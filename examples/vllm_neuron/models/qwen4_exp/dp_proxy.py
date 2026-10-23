# SPDX-License-Identifier: Apache-2.0
"""Least-outstanding-requests HTTP proxy over independent vLLM replicas (data parallel serving).

Each Qwen3.8-Flash-Next replica (serve.sh) runs on its own 8 NeuronCores and port; the replicas
share nothing but the page cache (checkpoint and n-gram table). This proxy exposes one
OpenAI-compatible endpoint and sends each request to the healthy replica with the fewest requests
in flight. Responses (including SSE streams) are relayed as they arrive. A background probe of
each replica's /health every few seconds decides which replicas are healthy, so replicas still
starting (or restarting) are skipped; requests wait while none is healthy.

Usage:
    python dp_proxy.py --port 8000 --backends http://127.0.0.1:8001,http://127.0.0.1:8002
"""

import argparse
import asyncio
import itertools
import logging

import aiohttp
from aiohttp import web

logger = logging.getLogger("dp_proxy")
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


class Proxy:
    def __init__(self, backends: list[str], timeout_s: float):
        self.backends = backends
        self.inflight = dict.fromkeys(backends, 0)
        self.healthy = dict.fromkeys(backends, False)
        self.served = dict.fromkeys(backends, 0)
        self._tiebreak = itertools.count()
        self.timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
        self.session: aiohttp.ClientSession | None = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout, connector=aiohttp.TCPConnector(limit=0)
        )
        self._probe_task = asyncio.create_task(self._probe())

    async def _probe(self):
        probe_timeout = aiohttp.ClientTimeout(total=5)
        while True:
            for b in self.backends:
                try:
                    async with self.session.get(
                        b + "/health", timeout=probe_timeout
                    ) as r:
                        ok = r.status == 200
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    ok = False
                if ok != self.healthy[b]:
                    logger.info("backend %s %s", b, "up" if ok else "down")
                self.healthy[b] = ok
            await asyncio.sleep(5)

    async def stop(self, app):
        self._probe_task.cancel()
        await self.session.close()

    def _pick(self) -> str | None:
        n = next(self._tiebreak)
        k = len(self.backends)
        # Fewest in flight among healthy replicas; rotate the start so ties spread evenly.
        order = [self.backends[(n + i) % k] for i in range(k)]
        order = [b for b in order if self.healthy[b]]
        return min(order, key=lambda b: self.inflight[b]) if order else None

    async def handle(self, request: web.Request) -> web.StreamResponse:
        backend = self._pick()
        while backend is None:  # no replica up yet
            await asyncio.sleep(2)
            backend = self._pick()
        self.inflight[backend] += 1
        self.served[backend] += 1
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
        }
        try:
            async with self.session.request(
                request.method,
                backend + request.rel_url.path_qs,
                data=body,
                headers=headers,
            ) as upstream:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in HOP_HEADERS
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("backend %s failed: %r", backend, e)
            return web.json_response(
                {"error": {"message": f"backend {backend} failed: {e!r}"}}, status=502
            )
        finally:
            self.inflight[backend] -= 1

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"healthy": self.healthy, "inflight": self.inflight, "served": self.served}
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backends", required=True, help="comma-separated base URLs")
    p.add_argument("--timeout-s", type=float, default=7200)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    proxy = Proxy([b.rstrip("/") for b in args.backends.split(",")], args.timeout_s)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_get("/proxy/stats", proxy.stats)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
