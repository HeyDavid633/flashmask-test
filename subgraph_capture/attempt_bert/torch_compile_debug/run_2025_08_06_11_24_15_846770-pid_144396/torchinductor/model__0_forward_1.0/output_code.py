# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
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


# kernel path: /tmp/torchinductor_root/tmpzurgvbnb/hr/chrc4oz7xxvlsj2kjq27om735zpmx2iql4yl52fitu4yi2ekvdpa.py
# Topologically Sorted Source Nodes: [input_embeds, position_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1, add_2, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
#   input_embeds => embedding
#   position_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %slice_2), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float16), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.full([RBLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30522), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r1 + 768*tmp4), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.full([RBLOCK], 512, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 512), "index out of bounds: 0 <= tmp11 < 512")
    tmp13 = tl.load(in_ptr3 + (r1 + 768*tmp11), rmask, other=0.0).to(tl.float32)
    tmp14 = tmp6 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = 768.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-12
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp15 - tmp25
    tmp38 = tmp37 * tmp36
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr0 + (r1 + 768*x0), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp36, None)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp45, rmask)
    tl.store(out_ptr1 + (x0), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpzurgvbnb/vc/cvcv4tla7xcju6ahyawnvmlz6qbl4votjoqfpwlj5qag4h3vxgpi.py
# Topologically Sorted Source Nodes: [expanded_mask, tensor, inverted_mask, to_1, attention_mask], Original ATen: [aten._to_copy, aten.lift_fresh, aten.sub, aten.masked_fill]
# Source node to ATen node mapping:
#   attention_mask => full_default_1, where
#   expanded_mask => convert_element_type_2
#   inverted_mask => sub_1
#   tensor => full_default
#   to_1 => convert_element_type_3
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand, torch.float16), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cpu, pin_memory: False})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %convert_element_type_2), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_1, torch.bool), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -65504.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_3, %full_default_1, %sub_1), kwargs = {})
triton_poi_fused__to_copy_lift_fresh_masked_fill_sub_1 = async_compile.triton('triton_poi_fused__to_copy_lift_fresh_masked_fill_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_lift_fresh_masked_fill_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_lift_fresh_masked_fill_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -65504.0
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpzurgvbnb/3l/c3lsk74tiyu6rx5axr4dzddh4g4eulse7rij4w4ddvlwx7q6tdi2.py
# Topologically Sorted Source Nodes: [add_1, sa_output], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_3
#   sa_output => add_4, add_5, convert_element_type_16, convert_element_type_17, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %convert_element_type_1), kwargs = {})
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_16, %getitem_7), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_16), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_17), kwargs = {})
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float16), kwargs = {})
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_per_fused_add_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp5 - tmp15
    tmp28 = tmp27 * tmp26
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp4, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr1 + (r1 + 768*x0), tmp35, rmask)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tmpzurgvbnb/hh/chhjtavhpxhsmdg5kztdxgfxhsj5vrrnitu2br2exxtjrchye3mt.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_1 => add_6, convert_element_type_21, convert_element_type_22, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %convert_element_type_21 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_21, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_21, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_6), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.float16), kwargs = {})
triton_poi_fused_gelu_3 = async_compile.triton('triton_poi_fused_gelu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103 = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (30522, 768), (768, 1))
    assert_size_stride(primals_3, (1, 512), (512, 1))
    assert_size_stride(primals_4, (512, 768), (768, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (1, 128), (128, 1))
    assert_size_stride(primals_8, (768, 768), (768, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, 768), (768, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (3072, 768), (768, 1))
    assert_size_stride(primals_19, (3072, ), (1, ))
    assert_size_stride(primals_20, (768, 3072), (3072, 1))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, 768), (768, 1))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, 768), (768, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, 768), (768, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (3072, 768), (768, 1))
    assert_size_stride(primals_35, (3072, ), (1, ))
    assert_size_stride(primals_36, (768, 3072), (3072, 1))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, 768), (768, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (3072, 768), (768, 1))
    assert_size_stride(primals_51, (3072, ), (1, ))
    assert_size_stride(primals_52, (768, 3072), (3072, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, 768), (768, 1))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, 768), (768, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (3072, 768), (768, 1))
    assert_size_stride(primals_67, (3072, ), (1, ))
    assert_size_stride(primals_68, (768, 3072), (3072, 1))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, 768), (768, 1))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (3072, 768), (768, 1))
    assert_size_stride(primals_83, (3072, ), (1, ))
    assert_size_stride(primals_84, (768, 3072), (3072, 1))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, 768), (768, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 768), (768, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, 768), (768, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (3072, 768), (768, 1))
    assert_size_stride(primals_99, (3072, ), (1, ))
    assert_size_stride(primals_100, (768, 3072), (3072, 1))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        buf1 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf4 = reinterpret_tensor(buf2, (1, 128, 1), (128, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_embeds, position_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(buf4, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, buf0, buf1, buf5, 128, 768, grid=grid(128), stream=stream0)
        del primals_2
        del primals_4
        del primals_6
        buf6 = empty_strided_cuda((1, 1, 128, 128), (16384, 16384, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [expanded_mask, tensor, inverted_mask, to_1, attention_mask], Original ATen: [aten._to_copy, aten.lift_fresh, aten.sub, aten.masked_fill]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_lift_fresh_masked_fill_sub_1.run(primals_7, buf6, 16384, grid=grid(16384), stream=stream0)
        del primals_7
        buf7 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf5, (128, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del primals_9
        buf8 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf5, (128, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
        del primals_11
        buf9 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf5, (128, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del primals_13
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf7, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf9, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf11 = buf10[0]
        buf12 = buf10[1]
        buf13 = buf10[2]
        buf14 = buf10[3]
        del buf10
        buf15 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (128, 768), (768, 1), 0), reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf15)
        buf16 = reinterpret_tensor(buf15, (1, 128, 768), (98304, 768, 1), 0); del buf15  # reuse
        buf17 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf20 = reinterpret_tensor(buf18, (1, 128, 1), (128, 1, 1), 0); del buf18  # reuse
        buf21 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_1, sa_output], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf16, buf20, primals_15, buf5, primals_16, primals_17, buf17, buf21, 128, 768, grid=grid(128), stream=stream0)
        del primals_15
        del primals_17
        buf22 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf21, (128, 768), (768, 1), 0), reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf22)
        del primals_19
        buf23 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf22, buf23, 393216, grid=grid(393216), stream=stream0)
        buf24 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf23, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf24)
        buf25 = reinterpret_tensor(buf24, (1, 128, 768), (98304, 768, 1), 0); del buf24  # reuse
        buf26 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf27 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf29 = reinterpret_tensor(buf27, (1, 128, 1), (128, 1, 1), 0); del buf27  # reuse
        buf30 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_2, ffn_output], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf25, buf29, primals_21, buf21, primals_22, primals_23, buf26, buf30, 128, 768, grid=grid(128), stream=stream0)
        del primals_21
        del primals_23
        buf31 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf30, (128, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
        del primals_25
        buf32 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf30, (128, 768), (768, 1), 0), reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del primals_27
        buf33 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf30, (128, 768), (768, 1), 0), reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
        del primals_29
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf31, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf32, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf33, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        buf37 = buf34[2]
        buf38 = buf34[3]
        del buf34
        buf39 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 768), (768, 1), 0), reinterpret_tensor(primals_30, (768, 768), (1, 768), 0), out=buf39)
        buf40 = reinterpret_tensor(buf39, (1, 128, 768), (98304, 768, 1), 0); del buf39  # reuse
        buf41 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf42 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf44 = reinterpret_tensor(buf42, (1, 128, 1), (128, 1, 1), 0); del buf42  # reuse
        buf45 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_3, sa_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf40, buf44, primals_31, buf30, primals_32, primals_33, buf41, buf45, 128, 768, grid=grid(128), stream=stream0)
        del primals_31
        del primals_33
        buf46 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, reinterpret_tensor(buf45, (128, 768), (768, 1), 0), reinterpret_tensor(primals_34, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf46)
        del primals_35
        buf47 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf46, buf47, 393216, grid=grid(393216), stream=stream0)
        buf48 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_36, (3072, 768), (1, 3072), 0), out=buf48)
        buf49 = reinterpret_tensor(buf48, (1, 128, 768), (98304, 768, 1), 0); del buf48  # reuse
        buf50 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf51 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf53 = reinterpret_tensor(buf51, (1, 128, 1), (128, 1, 1), 0); del buf51  # reuse
        buf54 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_4, ffn_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf49, buf53, primals_37, buf45, primals_38, primals_39, buf50, buf54, 128, 768, grid=grid(128), stream=stream0)
        del primals_37
        del primals_39
        buf55 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf54, (128, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf55)
        del primals_41
        buf56 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf54, (128, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf56)
        del primals_43
        buf57 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, reinterpret_tensor(buf54, (128, 768), (768, 1), 0), reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf57)
        del primals_45
        # Topologically Sorted Source Nodes: [attn_output_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf55, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf56, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf57, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf59 = buf58[0]
        buf60 = buf58[1]
        buf61 = buf58[2]
        buf62 = buf58[3]
        del buf58
        buf63 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf59, (128, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), out=buf63)
        buf64 = reinterpret_tensor(buf63, (1, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
        buf65 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf66 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf68 = reinterpret_tensor(buf66, (1, 128, 1), (128, 1, 1), 0); del buf66  # reuse
        buf69 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_5, sa_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf64, buf68, primals_47, buf54, primals_48, primals_49, buf65, buf69, 128, 768, grid=grid(128), stream=stream0)
        del primals_47
        del primals_49
        buf70 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(primals_50, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf70)
        del primals_51
        buf71 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf70, buf71, 393216, grid=grid(393216), stream=stream0)
        buf72 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf71, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_52, (3072, 768), (1, 3072), 0), out=buf72)
        buf73 = reinterpret_tensor(buf72, (1, 128, 768), (98304, 768, 1), 0); del buf72  # reuse
        buf74 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf75 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf77 = reinterpret_tensor(buf75, (1, 128, 1), (128, 1, 1), 0); del buf75  # reuse
        buf78 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_6, ffn_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf73, buf77, primals_53, buf69, primals_54, primals_55, buf74, buf78, 128, 768, grid=grid(128), stream=stream0)
        del primals_53
        del primals_55
        buf79 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, reinterpret_tensor(buf78, (128, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf79)
        del primals_57
        buf80 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, reinterpret_tensor(buf78, (128, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf80)
        del primals_59
        buf81 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, reinterpret_tensor(buf78, (128, 768), (768, 1), 0), reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf81)
        del primals_61
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf82 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf79, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf80, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf81, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf83 = buf82[0]
        buf84 = buf82[1]
        buf85 = buf82[2]
        buf86 = buf82[3]
        del buf82
        buf87 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf83, (128, 768), (768, 1), 0), reinterpret_tensor(primals_62, (768, 768), (1, 768), 0), out=buf87)
        buf88 = reinterpret_tensor(buf87, (1, 128, 768), (98304, 768, 1), 0); del buf87  # reuse
        buf89 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf90 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf92 = reinterpret_tensor(buf90, (1, 128, 1), (128, 1, 1), 0); del buf90  # reuse
        buf93 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_7, sa_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf88, buf92, primals_63, buf78, primals_64, primals_65, buf89, buf93, 128, 768, grid=grid(128), stream=stream0)
        del primals_63
        del primals_65
        buf94 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, reinterpret_tensor(buf93, (128, 768), (768, 1), 0), reinterpret_tensor(primals_66, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf94)
        del primals_67
        buf95 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf94, buf95, 393216, grid=grid(393216), stream=stream0)
        buf96 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf95, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_68, (3072, 768), (1, 3072), 0), out=buf96)
        buf97 = reinterpret_tensor(buf96, (1, 128, 768), (98304, 768, 1), 0); del buf96  # reuse
        buf98 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf99 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf101 = reinterpret_tensor(buf99, (1, 128, 1), (128, 1, 1), 0); del buf99  # reuse
        buf102 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_8, ffn_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf97, buf101, primals_69, buf93, primals_70, primals_71, buf98, buf102, 128, 768, grid=grid(128), stream=stream0)
        del primals_69
        del primals_71
        buf103 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf102, (128, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf103)
        del primals_73
        buf104 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, reinterpret_tensor(buf102, (128, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf104)
        del primals_75
        buf105 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, reinterpret_tensor(buf102, (128, 768), (768, 1), 0), reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
        del primals_77
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf106 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf103, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf104, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf105, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf107 = buf106[0]
        buf108 = buf106[1]
        buf109 = buf106[2]
        buf110 = buf106[3]
        del buf106
        buf111 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf107, (128, 768), (768, 1), 0), reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), out=buf111)
        buf112 = reinterpret_tensor(buf111, (1, 128, 768), (98304, 768, 1), 0); del buf111  # reuse
        buf113 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf114 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf116 = reinterpret_tensor(buf114, (1, 128, 1), (128, 1, 1), 0); del buf114  # reuse
        buf117 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_9, sa_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf112, buf116, primals_79, buf102, primals_80, primals_81, buf113, buf117, 128, 768, grid=grid(128), stream=stream0)
        del primals_79
        del primals_81
        buf118 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, reinterpret_tensor(buf117, (128, 768), (768, 1), 0), reinterpret_tensor(primals_82, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf118)
        del primals_83
        buf119 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf118, buf119, 393216, grid=grid(393216), stream=stream0)
        buf120 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf119, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_84, (3072, 768), (1, 3072), 0), out=buf120)
        buf121 = reinterpret_tensor(buf120, (1, 128, 768), (98304, 768, 1), 0); del buf120  # reuse
        buf122 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf123 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf125 = reinterpret_tensor(buf123, (1, 128, 1), (128, 1, 1), 0); del buf123  # reuse
        buf126 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_10, ffn_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf121, buf125, primals_85, buf117, primals_86, primals_87, buf122, buf126, 128, 768, grid=grid(128), stream=stream0)
        del primals_85
        del primals_87
        buf127 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_89, reinterpret_tensor(buf126, (128, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del primals_89
        buf128 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf126, (128, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf128)
        del primals_91
        buf129 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, reinterpret_tensor(buf126, (128, 768), (768, 1), 0), reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf129)
        del primals_93
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf130 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf127, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf128, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf129, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 128, 128), (16384, 0, 128, 1), 0), True)
        buf131 = buf130[0]
        buf132 = buf130[1]
        buf133 = buf130[2]
        buf134 = buf130[3]
        del buf130
        buf135 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 768), (1, 768), 0), out=buf135)
        buf136 = reinterpret_tensor(buf135, (1, 128, 768), (98304, 768, 1), 0); del buf135  # reuse
        buf137 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf138 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf140 = reinterpret_tensor(buf138, (1, 128, 1), (128, 1, 1), 0); del buf138  # reuse
        buf141 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_11, sa_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf136, buf140, primals_95, buf126, primals_96, primals_97, buf137, buf141, 128, 768, grid=grid(128), stream=stream0)
        del primals_95
        del primals_97
        buf142 = empty_strided_cuda((128, 3072), (3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, reinterpret_tensor(buf141, (128, 768), (768, 1), 0), reinterpret_tensor(primals_98, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf142)
        del primals_99
        buf143 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf142, buf143, 393216, grid=grid(393216), stream=stream0)
        buf144 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf143, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 768), (1, 3072), 0), out=buf144)
        buf145 = reinterpret_tensor(buf144, (1, 128, 768), (98304, 768, 1), 0); del buf144  # reuse
        buf146 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf147 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf149 = reinterpret_tensor(buf147, (1, 128, 1), (128, 1, 1), 0); del buf147  # reuse
        buf150 = empty_strided_cuda((1, 128, 768), (98304, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_12, ffn_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf145, buf149, primals_101, buf141, primals_102, primals_103, buf146, buf150, 128, 768, grid=grid(128), stream=stream0)
        del primals_101
        del primals_103
    return (buf150, primals_1, primals_5, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, reinterpret_tensor(primals_3, (1, 128), (512, 1), 0), buf0, buf1, buf4, buf6, reinterpret_tensor(buf5, (128, 768), (768, 1), 0), reinterpret_tensor(buf7, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf9, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf11, buf12, buf13, buf14, buf16, buf17, buf20, reinterpret_tensor(buf21, (128, 768), (768, 1), 0), buf22, reinterpret_tensor(buf23, (128, 3072), (3072, 1), 0), buf25, buf26, buf29, reinterpret_tensor(buf30, (128, 768), (768, 1), 0), reinterpret_tensor(buf31, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf32, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf33, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf35, buf36, buf37, buf38, buf40, buf41, buf44, reinterpret_tensor(buf45, (128, 768), (768, 1), 0), buf46, reinterpret_tensor(buf47, (128, 3072), (3072, 1), 0), buf49, buf50, buf53, reinterpret_tensor(buf54, (128, 768), (768, 1), 0), reinterpret_tensor(buf55, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf56, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf57, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf59, buf60, buf61, buf62, buf64, buf65, buf68, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), buf70, reinterpret_tensor(buf71, (128, 3072), (3072, 1), 0), buf73, buf74, buf77, reinterpret_tensor(buf78, (128, 768), (768, 1), 0), reinterpret_tensor(buf79, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf80, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf81, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf83, buf84, buf85, buf86, buf88, buf89, buf92, reinterpret_tensor(buf93, (128, 768), (768, 1), 0), buf94, reinterpret_tensor(buf95, (128, 3072), (3072, 1), 0), buf97, buf98, buf101, reinterpret_tensor(buf102, (128, 768), (768, 1), 0), reinterpret_tensor(buf103, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf104, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf105, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf107, buf108, buf109, buf110, buf112, buf113, buf116, reinterpret_tensor(buf117, (128, 768), (768, 1), 0), buf118, reinterpret_tensor(buf119, (128, 3072), (3072, 1), 0), buf121, buf122, buf125, reinterpret_tensor(buf126, (128, 768), (768, 1), 0), reinterpret_tensor(buf127, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf128, (1, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf129, (1, 12, 128, 64), (98304, 64, 768, 1), 0), buf131, buf132, buf133, buf134, buf136, buf137, buf140, reinterpret_tensor(buf141, (128, 768), (768, 1), 0), buf142, reinterpret_tensor(buf143, (128, 3072), (3072, 1), 0), buf145, buf146, buf149, primals_100, primals_98, primals_94, primals_92, primals_90, primals_88, primals_84, primals_82, primals_78, primals_76, primals_74, primals_72, primals_68, primals_66, primals_62, primals_60, primals_58, primals_56, primals_52, primals_50, primals_46, primals_44, primals_42, primals_40, primals_36, primals_34, primals_30, primals_28, primals_26, primals_24, primals_20, primals_18, primals_14, primals_12, primals_10, primals_8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_66 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_82 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_83 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_84 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_94 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_98 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_99 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_100 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
