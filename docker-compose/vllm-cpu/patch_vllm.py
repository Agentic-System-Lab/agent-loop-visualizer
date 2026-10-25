"""Two fixes to vLLM 0.32.0's CPU image, without which the models of the page fail on it.

1. vLLM's fix #61992 (https://github.com/vllm-project/vllm/pull/61992), which landed on vLLM's main branch the day
   after v0.32.0 was cut. On the CPU, vLLM runs a plain function in place of each Triton kernel and hands it the
   kernel's arguments in the kernel's order. In v0.32.0 the stand-in for the slot-mapping kernel takes a block_size
   the kernel does not have, so every argument after it arrives one place off, and every model's first request
   fails with "Boolean value of Tensor with more than one value is ambiguous".

2. One of our own, not yet in vLLM. On the CPU, a linear layer hands its input to oneDNN, which needs each row
   contiguous in memory. Gemma 4's audio encoder hands one a transposed tensor, so the first clip it hears ends in
   "Expected a.stride(-1) == 1" and takes the engine down with it. A contiguous copy of the input computes the same.

Each fix replaces code it finds exactly once, and the build fails if it finds something else: then a newer image may
not need it, or needs it written anew.
"""

import importlib.util
from pathlib import Path

VLLM = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])  # found without importing vllm and torch

FIXES = {
    "utils/cpu_triton_utils.py": ('''    block_table_stride: int,  # max_num_blocks_per_req
    block_size: int,
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: int | None = None,
    BLOCKS_PER_KV_BLOCK: int = 1,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = -1,
    BLOCK_SIZE: int = 1024,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    if BLOCKS_PER_KV_BLOCK != 1:
        assert block_size * BLOCKS_PER_KV_BLOCK == KV_CACHE_BLOCK_SIZE
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        block_size,
    )
''', '''    block_table_stride: int,  # max_num_blocks_per_req
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: int,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = -1,
    BLOCK_SIZE: int = 1024,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        KV_CACHE_BLOCK_SIZE,
    )
'''),
    "model_executor/layers/utils.py": ('''    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)
''', '''    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x if x.stride(-1) == 1 else x.contiguous(), weight, bias)
'''),
}

for name, (old, new) in FIXES.items():
    path = VLLM / name
    source = path.read_text()
    if source.count(old) != 1:
        raise SystemExit(f"{path} is not the one in vLLM 0.32.0's CPU image: see whether a newer image needs this fix at all.")
    path.write_text(source.replace(old, new))
    # The image ships compiled bytecode; drop the old code's, so that Python compiles the new.
    for compiled in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        compiled.unlink()
    print(f"Patched {path}")
