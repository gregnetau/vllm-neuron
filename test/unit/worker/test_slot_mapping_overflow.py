# SPDX-License-Identifier: Apache-2.0
"""Slot mapping of a recurrent-state group does not overflow int32 (CPU).

A state group's block size is max_model_len, so block_id * block_size passes 2**31 at long
context with a large pool; the GDN prefill reads slot_mapping > 0 as "real token".
"""

import numpy as np
import pytest

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")


def test_large_block_id_times_block_size_stays_positive():
    block_size = 262144
    block_table = np.array([[9000], [16384]], dtype=np.int32)
    positions = np.array([0, 5], dtype=np.int64)
    req_indices = np.array([0, 1], dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    runner_mod._compute_slot_mapping_cpu(
        block_table, out, positions, req_indices, block_size
    )
    assert out.tolist() == [9000 * block_size, 16384 * block_size + 5]
