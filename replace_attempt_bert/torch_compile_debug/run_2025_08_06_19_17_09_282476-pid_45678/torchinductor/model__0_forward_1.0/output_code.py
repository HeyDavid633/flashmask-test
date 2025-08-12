# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/bj/cbjy42sybkgmurslkgxoqnjxhlfilqzpbozxn4evudmqfcly7sdv.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_2 => add_2, add_3, convert_element_type_1, convert_element_type_2, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %view), kwargs = {})
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %unsqueeze), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float16), kwargs = {})
triton_per_fused_add_arange_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_arange_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr0': '*i64', 'out_ptr1': '*fp16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_arange_embedding_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_arange_embedding_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = x0
    tmp2 = tl.full([R0_BLOCK], 50257, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 50257), "index out of bounds: 0 <= tmp5 < 50257")
    tmp7 = tl.load(in_ptr1 + (r0_1 + 768*tmp5), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*tmp0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [R0_BLOCK])
    tmp25 = tl.where(r0_mask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = 768.0
    tmp28 = (tmp26 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp10 - tmp20
    tmp33 = tmp32 * tmp31
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 * tmp35
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp9, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp40, r0_mask)
    tl.store(out_ptr2 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/pe/cpeb663ttvl3jsykab3tbszuren5gfklcposl2v6wkg3drmtooew.py
# Topologically Sorted Source Nodes: [query], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   query => clone_1
# Graph fragment:
#   %clone_1 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 2304*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/hb/chb6ne3k72ujsx3a3vs34payk6krk3svillbkc7twp6aias4tlae.py
# Topologically Sorted Source Nodes: [key], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key => clone_2
# Graph fragment:
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 64*x2 + 2304*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (768 + x0 + 64*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/nd/cndddto7te4665xupcwlavowklmgddrajssek76e7pylzb6i5td5.py
# Topologically Sorted Source Nodes: [value], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   value => clone_3
# Graph fragment:
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 64*x2 + 2304*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 64*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/gp/cgpuwbdspegtedcm5hty4cmlhcdahtoinkipl2bm5ytqvhfke3ed.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   attn_output => full_default_1, full_default_2, where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
triton_poi_fused_scalar_tensor_where_4 = async_compile.triton('triton_poi_fused_scalar_tensor_where_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scalar_tensor_where_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp1 <= tmp0
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp6 = (tmp5 != 0)
    tmp7 = tmp4 & tmp6
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/p4/cp4s4mrzpfd4vxuzq2x6d5sisouwqwlq73q7r5xpgbrl22famy7h.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_3 => add_4
#   hidden_states_4 => add_5, add_6, convert_element_type_10, convert_element_type_9, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %getitem_10), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_12), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float16), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 768.0
    tmp23 = (tmp21 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp5 - tmp15
    tmp28 = tmp27 * tmp26
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp4, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/j6/cj6opp4hodvsd7kfhtmth6rrlely4eetct4atvzsefpizx3c4skj.py
# Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_11], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
# Source node to ATen node mapping:
#   add_2 => add_7
#   add_3 => add_8
#   hidden_states_5 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   view_11 => view_17
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.5), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_16, 3.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_8), kwargs = {})
#   %view_17 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [-1, 3072]), kwargs = {})
triton_poi_fused_add_mul_pow_tanh_view_6 = async_compile.triton('triton_poi_fused_add_mul_pow_tanh_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = 0.044715
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = 0.7978845608028654
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tl.store(out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpuqp_iipx/fy/cfyzo7bh2v6fhnrxshwk66o3z6mpgsm2vkwh3kigosx6rvcghzl5.py
# Topologically Sorted Source Nodes: [hidden_states_7, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => add_9
#   hidden_states_8 => add_10, add_11, convert_element_type_17, convert_element_type_18, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_18), kwargs = {})
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_17, %getitem_12), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_17), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_18), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float16), kwargs = {})
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_per_fused_add_native_layer_norm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '9772640631AE68FEA022A8EF6618407D4C87E85AB83EA243AE168DD2AF8BEE7A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 768.0
    tmp23 = (tmp21 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp5 - tmp15
    tmp28 = tmp27 * tmp26
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp4, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78 = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (50257, 768), (768, 1))
    assert_size_stride(primals_3, (1024, 768), (768, 1))
    assert_size_stride(primals_4, (1, 128), (128, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (2304, ), (1, ))
    assert_size_stride(primals_8, (768, 2304), (2304, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (3072, ), (1, ))
    assert_size_stride(primals_14, (768, 3072), (3072, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (3072, 768), (768, 1))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (2304, ), (1, ))
    assert_size_stride(primals_20, (768, 2304), (2304, 1))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (3072, ), (1, ))
    assert_size_stride(primals_26, (768, 3072), (3072, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (3072, 768), (768, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (2304, ), (1, ))
    assert_size_stride(primals_32, (768, 2304), (2304, 1))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, 768), (768, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (3072, ), (1, ))
    assert_size_stride(primals_38, (768, 3072), (3072, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (3072, 768), (768, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (2304, ), (1, ))
    assert_size_stride(primals_44, (768, 2304), (2304, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (3072, ), (1, ))
    assert_size_stride(primals_50, (768, 3072), (3072, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (3072, 768), (768, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (2304, ), (1, ))
    assert_size_stride(primals_56, (768, 2304), (2304, 1))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (3072, ), (1, ))
    assert_size_stride(primals_62, (768, 3072), (3072, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (3072, 768), (768, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (2304, ), (1, ))
    assert_size_stride(primals_68, (768, 2304), (2304, 1))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (3072, ), (1, ))
    assert_size_stride(primals_74, (768, 3072), (3072, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (3072, 768), (768, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, ), (1, ), torch.int64)
        buf1 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        buf2 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf5 = reinterpret_tensor(buf3, (1, 128, 1), (128, 1, 1), 0); del buf3  # reuse
        buf6 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_arange_embedding_native_layer_norm_0.run(buf5, primals_1, primals_2, primals_3, primals_5, primals_6, buf0, buf1, buf2, buf6, 128, 768, stream=stream0)
        del primals_2
        del primals_3
        del primals_6
        buf7 = empty_strided_cuda((128, 2304), (2304, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 768), (768, 1), 0), primals_8, out=buf7)
        buf8 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf7, primals_7, buf8, 98304, stream=stream0)
        buf9 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf7, primals_7, buf9, 98304, stream=stream0)
        buf10 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf7, primals_7, buf10, 98304, stream=stream0)
        del primals_7
        buf11 = empty_strided_cuda((1, 1, 128, 128), (16384, 16384, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.scalar_tensor, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scalar_tensor_where_4.run(buf0, primals_4, buf11, 16384, stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf8, buf9, buf10, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf13 = buf12[0]
        assert_size_stride(buf13, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf14 = buf12[1]
        assert_size_stride(buf14, (1, 12, 128), (1536, 128, 1))
        buf15 = buf12[2]
        assert_size_stride(buf15, (), ())
        buf16 = buf12[3]
        assert_size_stride(buf16, (), ())
        del buf12
        buf17 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf13, (128, 768), (768, 1), 0), primals_10, out=buf17)
        buf18 = reinterpret_tensor(buf17, (1, 128, 768), (98304, 768, 1), 0); del buf17  # reuse
        buf19 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf20 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf22 = reinterpret_tensor(buf20, (1, 128, 1), (128, 1, 1), 0); del buf20  # reuse
        buf23 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf18, buf22, primals_9, buf1, primals_11, primals_12, buf19, buf23, 128, 768, stream=stream0)
        del primals_12
        del primals_9
        buf24 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf23, (128, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf24)
        del primals_13
        buf25 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_11], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf24, buf25, 393216, stream=stream0)
        buf26 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf25, primals_16, out=buf26)
        buf27 = reinterpret_tensor(buf26, (1, 128, 768), (98304, 768, 1), 0); del buf26  # reuse
        buf28 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf29 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf31 = reinterpret_tensor(buf29, (1, 128, 1), (128, 1, 1), 0); del buf29  # reuse
        buf32 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_7, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf27, buf31, buf18, primals_15, primals_17, primals_18, buf28, buf32, 128, 768, stream=stream0)
        del primals_15
        del primals_18
        buf33 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf32, (128, 768), (768, 1), 0), primals_20, out=buf33)
        buf34 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf33, primals_19, buf34, 98304, stream=stream0)
        buf35 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf33, primals_19, buf35, 98304, stream=stream0)
        buf36 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf33, primals_19, buf36, 98304, stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [attn_output_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf37 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf34, buf35, buf36, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf38 = buf37[0]
        assert_size_stride(buf38, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf39 = buf37[1]
        assert_size_stride(buf39, (1, 12, 128), (1536, 128, 1))
        buf40 = buf37[2]
        assert_size_stride(buf40, (), ())
        buf41 = buf37[3]
        assert_size_stride(buf41, (), ())
        del buf37
        buf42 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (128, 768), (768, 1), 0), primals_22, out=buf42)
        buf43 = reinterpret_tensor(buf42, (1, 128, 768), (98304, 768, 1), 0); del buf42  # reuse
        buf44 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf45 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf47 = reinterpret_tensor(buf45, (1, 128, 1), (128, 1, 1), 0); del buf45  # reuse
        buf48 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_9, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf43, buf47, primals_21, buf27, primals_23, primals_24, buf44, buf48, 128, 768, stream=stream0)
        del primals_21
        del primals_24
        buf49 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf48, (128, 768), (768, 1), 0), primals_26, alpha=1, beta=1, out=buf49)
        del primals_25
        buf50 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11, view_22], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf49, buf50, 393216, stream=stream0)
        buf51 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf50, primals_28, out=buf51)
        buf52 = reinterpret_tensor(buf51, (1, 128, 768), (98304, 768, 1), 0); del buf51  # reuse
        buf53 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf54 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf56 = reinterpret_tensor(buf54, (1, 128, 1), (128, 1, 1), 0); del buf54  # reuse
        buf57 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf52, buf56, buf43, primals_27, primals_29, primals_30, buf53, buf57, 128, 768, stream=stream0)
        del primals_27
        del primals_30
        buf58 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf57, (128, 768), (768, 1), 0), primals_32, out=buf58)
        buf59 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf58, primals_31, buf59, 98304, stream=stream0)
        buf60 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf58, primals_31, buf60, 98304, stream=stream0)
        buf61 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf58, primals_31, buf61, 98304, stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf62 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf59, buf60, buf61, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf63 = buf62[0]
        assert_size_stride(buf63, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf64 = buf62[1]
        assert_size_stride(buf64, (1, 12, 128), (1536, 128, 1))
        buf65 = buf62[2]
        assert_size_stride(buf65, (), ())
        buf66 = buf62[3]
        assert_size_stride(buf66, (), ())
        del buf62
        buf67 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf63, (128, 768), (768, 1), 0), primals_34, out=buf67)
        buf68 = reinterpret_tensor(buf67, (1, 128, 768), (98304, 768, 1), 0); del buf67  # reuse
        buf69 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf70 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf72 = reinterpret_tensor(buf70, (1, 128, 1), (128, 1, 1), 0); del buf70  # reuse
        buf73 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_15, hidden_states_16], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf68, buf72, primals_33, buf52, primals_35, primals_36, buf69, buf73, 128, 768, stream=stream0)
        del primals_33
        del primals_36
        buf74 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, reinterpret_tensor(buf73, (128, 768), (768, 1), 0), primals_38, alpha=1, beta=1, out=buf74)
        del primals_37
        buf75 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17, view_33], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf74, buf75, 393216, stream=stream0)
        buf76 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf75, primals_40, out=buf76)
        buf77 = reinterpret_tensor(buf76, (1, 128, 768), (98304, 768, 1), 0); del buf76  # reuse
        buf78 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf79 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf81 = reinterpret_tensor(buf79, (1, 128, 1), (128, 1, 1), 0); del buf79  # reuse
        buf82 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf77, buf81, buf68, primals_39, primals_41, primals_42, buf78, buf82, 128, 768, stream=stream0)
        del primals_39
        del primals_42
        buf83 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf82, (128, 768), (768, 1), 0), primals_44, out=buf83)
        buf84 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf83, primals_43, buf84, 98304, stream=stream0)
        buf85 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf83, primals_43, buf85, 98304, stream=stream0)
        buf86 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf83, primals_43, buf86, 98304, stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf87 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf84, buf85, buf86, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf88 = buf87[0]
        assert_size_stride(buf88, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf89 = buf87[1]
        assert_size_stride(buf89, (1, 12, 128), (1536, 128, 1))
        buf90 = buf87[2]
        assert_size_stride(buf90, (), ())
        buf91 = buf87[3]
        assert_size_stride(buf91, (), ())
        del buf87
        buf92 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf88, (128, 768), (768, 1), 0), primals_46, out=buf92)
        buf93 = reinterpret_tensor(buf92, (1, 128, 768), (98304, 768, 1), 0); del buf92  # reuse
        buf94 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf95 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf97 = reinterpret_tensor(buf95, (1, 128, 1), (128, 1, 1), 0); del buf95  # reuse
        buf98 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_22], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf93, buf97, primals_45, buf77, primals_47, primals_48, buf94, buf98, 128, 768, stream=stream0)
        del primals_45
        del primals_48
        buf99 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, reinterpret_tensor(buf98, (128, 768), (768, 1), 0), primals_50, alpha=1, beta=1, out=buf99)
        del primals_49
        buf100 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23, view_44], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf99, buf100, 393216, stream=stream0)
        buf101 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf100, primals_52, out=buf101)
        buf102 = reinterpret_tensor(buf101, (1, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
        buf103 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf104 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf106 = reinterpret_tensor(buf104, (1, 128, 1), (128, 1, 1), 0); del buf104  # reuse
        buf107 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf102, buf106, buf93, primals_51, primals_53, primals_54, buf103, buf107, 128, 768, stream=stream0)
        del primals_51
        del primals_54
        buf108 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf107, (128, 768), (768, 1), 0), primals_56, out=buf108)
        buf109 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf108, primals_55, buf109, 98304, stream=stream0)
        buf110 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf108, primals_55, buf110, 98304, stream=stream0)
        buf111 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf108, primals_55, buf111, 98304, stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [attn_output_16], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf112 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf109, buf110, buf111, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf113 = buf112[0]
        assert_size_stride(buf113, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf114 = buf112[1]
        assert_size_stride(buf114, (1, 12, 128), (1536, 128, 1))
        buf115 = buf112[2]
        assert_size_stride(buf115, (), ())
        buf116 = buf112[3]
        assert_size_stride(buf116, (), ())
        del buf112
        buf117 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 768), (768, 1), 0), primals_58, out=buf117)
        buf118 = reinterpret_tensor(buf117, (1, 128, 768), (98304, 768, 1), 0); del buf117  # reuse
        buf119 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf120 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf122 = reinterpret_tensor(buf120, (1, 128, 1), (128, 1, 1), 0); del buf120  # reuse
        buf123 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf118, buf122, primals_57, buf102, primals_59, primals_60, buf119, buf123, 128, 768, stream=stream0)
        del primals_57
        del primals_60
        buf124 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, reinterpret_tensor(buf123, (128, 768), (768, 1), 0), primals_62, alpha=1, beta=1, out=buf124)
        del primals_61
        buf125 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29, view_55], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf124, buf125, 393216, stream=stream0)
        buf126 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf125, primals_64, out=buf126)
        buf127 = reinterpret_tensor(buf126, (1, 128, 768), (98304, 768, 1), 0); del buf126  # reuse
        buf128 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf129 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf131 = reinterpret_tensor(buf129, (1, 128, 1), (128, 1, 1), 0); del buf129  # reuse
        buf132 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf127, buf131, buf118, primals_63, primals_65, primals_66, buf128, buf132, 128, 768, stream=stream0)
        del primals_63
        del primals_66
        buf133 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf132, (128, 768), (768, 1), 0), primals_68, out=buf133)
        buf134 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf133, primals_67, buf134, 98304, stream=stream0)
        buf135 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf133, primals_67, buf135, 98304, stream=stream0)
        buf136 = empty_strided_cuda((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf133, primals_67, buf136, 98304, stream=stream0)
        del buf133
        del primals_67
        # Topologically Sorted Source Nodes: [attn_output_20], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf137 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf134, buf135, buf136, reinterpret_tensor(buf11, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf138 = buf137[0]
        assert_size_stride(buf138, (1, 12, 128, 64), (98304, 64, 768, 1))
        buf139 = buf137[1]
        assert_size_stride(buf139, (1, 12, 128), (1536, 128, 1))
        buf140 = buf137[2]
        assert_size_stride(buf140, (), ())
        buf141 = buf137[3]
        assert_size_stride(buf141, (), ())
        del buf137
        buf142 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 768), (768, 1), 0), primals_70, out=buf142)
        buf143 = reinterpret_tensor(buf142, (1, 128, 768), (98304, 768, 1), 0); del buf142  # reuse
        buf144 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf145 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf147 = reinterpret_tensor(buf145, (1, 128, 1), (128, 1, 1), 0); del buf145  # reuse
        buf148 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_5.run(buf143, buf147, primals_69, buf127, primals_71, primals_72, buf144, buf148, 128, 768, stream=stream0)
        del primals_69
        del primals_72
        buf149 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf148, (128, 768), (768, 1), 0), primals_74, alpha=1, beta=1, out=buf149)
        del primals_73
        buf150 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35, view_66], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf149, buf150, 393216, stream=stream0)
        buf151 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf150, primals_76, out=buf151)
        buf152 = reinterpret_tensor(buf151, (1, 128, 768), (98304, 768, 1), 0); del buf151  # reuse
        buf153 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf154 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf156 = reinterpret_tensor(buf154, (1, 128, 1), (128, 1, 1), 0); del buf154  # reuse
        buf157 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf152, buf156, buf143, primals_75, primals_77, primals_78, buf153, buf157, 128, 768, stream=stream0)
        del primals_75
        del primals_78
    return (buf157, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_1, reinterpret_tensor(buf0, (1, 128), (128, 1), 0), buf1, buf2, buf5, buf8, buf9, buf10, buf11, buf13, buf14, buf15, buf16, buf18, buf19, buf22, buf24, buf27, buf28, buf31, buf34, buf35, buf36, buf38, buf39, buf40, buf41, buf43, buf44, buf47, buf49, buf52, buf53, buf56, buf59, buf60, buf61, buf63, buf64, buf65, buf66, buf68, buf69, buf72, buf74, buf77, buf78, buf81, buf84, buf85, buf86, buf88, buf89, buf90, buf91, buf93, buf94, buf97, buf99, buf102, buf103, buf106, buf109, buf110, buf111, buf113, buf114, buf115, buf116, buf118, buf119, buf122, buf124, buf127, buf128, buf131, buf134, buf135, buf136, buf138, buf139, buf140, buf141, buf143, buf144, buf147, buf149, buf152, buf153, buf156, reinterpret_tensor(primals_76, (768, 3072), (1, 768), 0), reinterpret_tensor(buf150, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_74, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf148, (768, 128), (1, 768), 0), reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), reinterpret_tensor(primals_68, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf132, (768, 128), (1, 768), 0), reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), reinterpret_tensor(buf125, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_62, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf123, (768, 128), (1, 768), 0), reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), reinterpret_tensor(primals_56, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf107, (768, 128), (1, 768), 0), reinterpret_tensor(primals_52, (768, 3072), (1, 768), 0), reinterpret_tensor(buf100, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf98, (768, 128), (1, 768), 0), reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), reinterpret_tensor(primals_44, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf82, (768, 128), (1, 768), 0), reinterpret_tensor(primals_40, (768, 3072), (1, 768), 0), reinterpret_tensor(buf75, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf73, (768, 128), (1, 768), 0), reinterpret_tensor(primals_34, (768, 768), (1, 768), 0), reinterpret_tensor(primals_32, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf57, (768, 128), (1, 768), 0), reinterpret_tensor(primals_28, (768, 3072), (1, 768), 0), reinterpret_tensor(buf50, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_26, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf48, (768, 128), (1, 768), 0), reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), reinterpret_tensor(primals_20, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf32, (768, 128), (1, 768), 0), reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), reinterpret_tensor(buf25, (3072, 128), (1, 3072), 0), reinterpret_tensor(primals_14, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf23, (768, 128), (1, 768), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), reinterpret_tensor(primals_8, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf6, (768, 128), (1, 768), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_3 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_61 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_73 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_76 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
