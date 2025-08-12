
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_level', 'repro_after', 'constant_functions', 'skipfiles_inline_module_allowlist'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu118
# torch cuda version: 11.8
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Sep_21_10:33:58_PDT_2022 
# Cuda compilation tools, release 11.8, V11.8.89 
# Build cuda_11.8.r11.8/compiler.31833905_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103):
        embedding = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = None
        slice_2 = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_4, slice_2);  primals_4 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_7, 1);  primals_7 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand = torch.ops.aten.expand.default(unsqueeze_1, [1, 1, 128, 128]);  unsqueeze_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(expand, torch.float16);  expand = None
        full_default = torch.ops.aten.full.default([], 1.0, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1 = torch.ops.aten.sub.Tensor(full_default, convert_element_type_2);  full_default = convert_element_type_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1 = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(convert_element_type_3, full_default_1, sub_1);  convert_element_type_3 = full_default_1 = sub_1 = None
        view = torch.ops.aten.view.default(convert_element_type_1, [128, 768])
        permute = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm = torch.ops.aten.addmm.default(primals_9, view, permute);  primals_9 = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 128, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_11, view, permute_2);  primals_11 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_4 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_13, view, permute_4);  primals_13 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [1, 128, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        expand_1 = torch.ops.aten.expand.default(where, [1, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_1, True)
        getitem_2 = _scaled_dot_product_efficient_attention[0]
        getitem_3 = _scaled_dot_product_efficient_attention[1]
        getitem_4 = _scaled_dot_product_efficient_attention[2]
        getitem_5 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])
        view_9 = torch.ops.aten.view.default(permute_6, [1, -1, 768]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(view_9, [128, 768]);  view_9 = None
        permute_7 = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_15, view_10, permute_7);  primals_15 = view_10 = None
        view_11 = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(view_11, convert_element_type_1);  view_11 = convert_element_type_1 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_7);  convert_element_type_16 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_16);  mul_2 = None
        add_5 = torch.ops.aten.add.Tensor(mul_3, primals_17);  mul_3 = primals_17 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None
        view_12 = torch.ops.aten.view.default(convert_element_type_17, [128, 768])
        permute_8 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        addmm_4 = torch.ops.aten.addmm.default(primals_19, view_12, permute_8);  primals_19 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [1, 128, 3072])
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_22, [128, 3072]);  convert_element_type_22 = None
        permute_9 = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_21, view_14, permute_9);  primals_21 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
        add_7 = torch.ops.aten.add.Tensor(view_15, convert_element_type_17);  view_15 = convert_element_type_17 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_9);  convert_element_type_26 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, primals_22);  mul_7 = None
        add_9 = torch.ops.aten.add.Tensor(mul_8, primals_23);  mul_8 = primals_23 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_27, [128, 768])
        permute_10 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_25, view_16, permute_10);  primals_25 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [1, 128, 768]);  addmm_6 = None
        view_18 = torch.ops.aten.view.default(view_17, [1, -1, 12, 64]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        permute_12 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_27, view_16, permute_12);  primals_27 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [1, -1, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        permute_14 = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_29, view_16, permute_14);  primals_29 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [1, 128, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [1, -1, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_1, True)
        getitem_10 = _scaled_dot_product_efficient_attention_1[0]
        getitem_11 = _scaled_dot_product_efficient_attention_1[1]
        getitem_12 = _scaled_dot_product_efficient_attention_1[2]
        getitem_13 = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3])
        view_25 = torch.ops.aten.view.default(permute_16, [1, -1, 768]);  permute_16 = None
        view_26 = torch.ops.aten.view.default(view_25, [128, 768]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_31, view_26, permute_17);  primals_31 = view_26 = None
        view_27 = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
        add_10 = torch.ops.aten.add.Tensor(view_27, convert_element_type_27);  view_27 = convert_element_type_27 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_15);  convert_element_type_40 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_32);  mul_9 = None
        add_12 = torch.ops.aten.add.Tensor(mul_10, primals_33);  mul_10 = primals_33 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None
        view_28 = torch.ops.aten.view.default(convert_element_type_41, [128, 768])
        permute_18 = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_35, view_28, permute_18);  primals_35 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [1, 128, 3072])
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.7071067811865476);  convert_element_type_45 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None
        view_30 = torch.ops.aten.view.default(convert_element_type_46, [128, 3072]);  convert_element_type_46 = None
        permute_19 = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_37, view_30, permute_19);  primals_37 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(view_31, convert_element_type_41);  view_31 = convert_element_type_41 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_17);  convert_element_type_50 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, primals_38);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(mul_15, primals_39);  mul_15 = primals_39 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_51, [128, 768])
        permute_20 = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_41, view_32, permute_20);  primals_41 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [1, 128, 768]);  addmm_12 = None
        view_34 = torch.ops.aten.view.default(view_33, [1, -1, 12, 64]);  view_33 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        permute_22 = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_43, view_32, permute_22);  primals_43 = None
        view_36 = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
        view_37 = torch.ops.aten.view.default(view_36, [1, -1, 12, 64]);  view_36 = None
        permute_23 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        permute_24 = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_45, view_32, permute_24);  primals_45 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [1, 128, 768]);  addmm_14 = None
        view_40 = torch.ops.aten.view.default(view_39, [1, -1, 12, 64]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_1, True)
        getitem_18 = _scaled_dot_product_efficient_attention_2[0]
        getitem_19 = _scaled_dot_product_efficient_attention_2[1]
        getitem_20 = _scaled_dot_product_efficient_attention_2[2]
        getitem_21 = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])
        view_41 = torch.ops.aten.view.default(permute_26, [1, -1, 768]);  permute_26 = None
        view_42 = torch.ops.aten.view.default(view_41, [128, 768]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_47, view_42, permute_27);  primals_47 = view_42 = None
        view_43 = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(view_43, convert_element_type_51);  view_43 = convert_element_type_51 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_23);  convert_element_type_64 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_48);  mul_16 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, primals_49);  mul_17 = primals_49 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_65, [128, 768])
        permute_28 = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_51, view_44, permute_28);  primals_51 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [1, 128, 3072])
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.7071067811865476);  convert_element_type_69 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None
        view_46 = torch.ops.aten.view.default(convert_element_type_70, [128, 3072]);  convert_element_type_70 = None
        permute_29 = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_53, view_46, permute_29);  primals_53 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
        add_21 = torch.ops.aten.add.Tensor(view_47, convert_element_type_65);  view_47 = convert_element_type_65 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_74, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_7 = torch.ops.aten.sub.Tensor(convert_element_type_74, getitem_25);  convert_element_type_74 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, primals_54);  mul_21 = None
        add_23 = torch.ops.aten.add.Tensor(mul_22, primals_55);  mul_22 = primals_55 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_75, [128, 768])
        permute_30 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_57, view_48, permute_30);  primals_57 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [1, 128, 768]);  addmm_18 = None
        view_50 = torch.ops.aten.view.default(view_49, [1, -1, 12, 64]);  view_49 = None
        permute_31 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_32 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_59, view_48, permute_32);  primals_59 = None
        view_52 = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
        view_53 = torch.ops.aten.view.default(view_52, [1, -1, 12, 64]);  view_52 = None
        permute_33 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_34 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_61, view_48, permute_34);  primals_61 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [1, 128, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(view_55, [1, -1, 12, 64]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_1, True)
        getitem_26 = _scaled_dot_product_efficient_attention_3[0]
        getitem_27 = _scaled_dot_product_efficient_attention_3[1]
        getitem_28 = _scaled_dot_product_efficient_attention_3[2]
        getitem_29 = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3])
        view_57 = torch.ops.aten.view.default(permute_36, [1, -1, 768]);  permute_36 = None
        view_58 = torch.ops.aten.view.default(view_57, [128, 768]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_63, view_58, permute_37);  primals_63 = view_58 = None
        view_59 = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
        add_24 = torch.ops.aten.add.Tensor(view_59, convert_element_type_75);  view_59 = convert_element_type_75 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8 = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_31);  convert_element_type_88 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, primals_64);  mul_23 = None
        add_26 = torch.ops.aten.add.Tensor(mul_24, primals_65);  mul_24 = primals_65 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None
        view_60 = torch.ops.aten.view.default(convert_element_type_89, [128, 768])
        permute_38 = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_67, view_60, permute_38);  primals_67 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [1, 128, 3072])
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.7071067811865476);  convert_element_type_93 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None
        view_62 = torch.ops.aten.view.default(convert_element_type_94, [128, 3072]);  convert_element_type_94 = None
        permute_39 = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_69, view_62, permute_39);  primals_69 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(view_63, convert_element_type_89);  view_63 = convert_element_type_89 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9 = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_33);  convert_element_type_98 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, primals_70);  mul_28 = None
        add_30 = torch.ops.aten.add.Tensor(mul_29, primals_71);  mul_29 = primals_71 = None
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None
        view_64 = torch.ops.aten.view.default(convert_element_type_99, [128, 768])
        permute_40 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_24 = torch.ops.aten.addmm.default(primals_73, view_64, permute_40);  primals_73 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [1, 128, 768]);  addmm_24 = None
        view_66 = torch.ops.aten.view.default(view_65, [1, -1, 12, 64]);  view_65 = None
        permute_41 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        permute_42 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        addmm_25 = torch.ops.aten.addmm.default(primals_75, view_64, permute_42);  primals_75 = None
        view_68 = torch.ops.aten.view.default(addmm_25, [1, 128, 768]);  addmm_25 = None
        view_69 = torch.ops.aten.view.default(view_68, [1, -1, 12, 64]);  view_68 = None
        permute_43 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        permute_44 = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        addmm_26 = torch.ops.aten.addmm.default(primals_77, view_64, permute_44);  primals_77 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [1, 128, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_71, [1, -1, 12, 64]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_1, True)
        getitem_34 = _scaled_dot_product_efficient_attention_4[0]
        getitem_35 = _scaled_dot_product_efficient_attention_4[1]
        getitem_36 = _scaled_dot_product_efficient_attention_4[2]
        getitem_37 = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3])
        view_73 = torch.ops.aten.view.default(permute_46, [1, -1, 768]);  permute_46 = None
        view_74 = torch.ops.aten.view.default(view_73, [128, 768]);  view_73 = None
        permute_47 = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        addmm_27 = torch.ops.aten.addmm.default(primals_79, view_74, permute_47);  primals_79 = view_74 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [1, 128, 768]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(view_75, convert_element_type_99);  view_75 = convert_element_type_99 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_10 = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_39);  convert_element_type_112 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, primals_80);  mul_30 = None
        add_33 = torch.ops.aten.add.Tensor(mul_31, primals_81);  mul_31 = primals_81 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None
        view_76 = torch.ops.aten.view.default(convert_element_type_113, [128, 768])
        permute_48 = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        addmm_28 = torch.ops.aten.addmm.default(primals_83, view_76, permute_48);  primals_83 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [1, 128, 3072])
        convert_element_type_117 = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.7071067811865476);  convert_element_type_117 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        convert_element_type_118 = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None
        view_78 = torch.ops.aten.view.default(convert_element_type_118, [128, 3072]);  convert_element_type_118 = None
        permute_49 = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        addmm_29 = torch.ops.aten.addmm.default(primals_85, view_78, permute_49);  primals_85 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [1, 128, 768]);  addmm_29 = None
        add_35 = torch.ops.aten.add.Tensor(view_79, convert_element_type_113);  view_79 = convert_element_type_113 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_122, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_11 = torch.ops.aten.sub.Tensor(convert_element_type_122, getitem_41);  convert_element_type_122 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, primals_86);  mul_35 = None
        add_37 = torch.ops.aten.add.Tensor(mul_36, primals_87);  mul_36 = primals_87 = None
        convert_element_type_123 = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None
        view_80 = torch.ops.aten.view.default(convert_element_type_123, [128, 768])
        permute_50 = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        addmm_30 = torch.ops.aten.addmm.default(primals_89, view_80, permute_50);  primals_89 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [1, 128, 768]);  addmm_30 = None
        view_82 = torch.ops.aten.view.default(view_81, [1, -1, 12, 64]);  view_81 = None
        permute_51 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        permute_52 = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        addmm_31 = torch.ops.aten.addmm.default(primals_91, view_80, permute_52);  primals_91 = None
        view_84 = torch.ops.aten.view.default(addmm_31, [1, 128, 768]);  addmm_31 = None
        view_85 = torch.ops.aten.view.default(view_84, [1, -1, 12, 64]);  view_84 = None
        permute_53 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        permute_54 = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        addmm_32 = torch.ops.aten.addmm.default(primals_93, view_80, permute_54);  primals_93 = None
        view_87 = torch.ops.aten.view.default(addmm_32, [1, 128, 768]);  addmm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [1, -1, 12, 64]);  view_87 = None
        permute_55 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_1, True);  expand_1 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0]
        getitem_43 = _scaled_dot_product_efficient_attention_5[1]
        getitem_44 = _scaled_dot_product_efficient_attention_5[2]
        getitem_45 = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3])
        view_89 = torch.ops.aten.view.default(permute_56, [1, -1, 768]);  permute_56 = None
        view_90 = torch.ops.aten.view.default(view_89, [128, 768]);  view_89 = None
        permute_57 = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        addmm_33 = torch.ops.aten.addmm.default(primals_95, view_90, permute_57);  primals_95 = view_90 = None
        view_91 = torch.ops.aten.view.default(addmm_33, [1, 128, 768]);  addmm_33 = None
        add_38 = torch.ops.aten.add.Tensor(view_91, convert_element_type_123);  view_91 = convert_element_type_123 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_12 = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_47);  convert_element_type_136 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, primals_96);  mul_37 = None
        add_40 = torch.ops.aten.add.Tensor(mul_38, primals_97);  mul_38 = primals_97 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_137, [128, 768])
        permute_58 = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        addmm_34 = torch.ops.aten.addmm.default(primals_99, view_92, permute_58);  primals_99 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [1, 128, 3072])
        convert_element_type_141 = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.7071067811865476);  convert_element_type_141 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None
        view_94 = torch.ops.aten.view.default(convert_element_type_142, [128, 3072]);  convert_element_type_142 = None
        permute_59 = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        addmm_35 = torch.ops.aten.addmm.default(primals_101, view_94, permute_59);  primals_101 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [1, 128, 768]);  addmm_35 = None
        add_42 = torch.ops.aten.add.Tensor(view_95, convert_element_type_137);  view_95 = convert_element_type_137 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_146, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_13 = torch.ops.aten.sub.Tensor(convert_element_type_146, getitem_49);  convert_element_type_146 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_102);  mul_42 = None
        add_44 = torch.ops.aten.add.Tensor(mul_43, primals_103);  mul_43 = primals_103 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None
        permute_60 = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_64 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_68 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_74 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_79 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_84 = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        permute_88 = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_92 = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_96 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_102 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_107 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_112 = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        permute_116 = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_120 = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_124 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_130 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_135 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_140 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_144 = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        permute_148 = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        permute_152 = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_158 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_163 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_168 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_172 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_176 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        permute_180 = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        permute_186 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_191 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_196 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_200 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_204 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_208 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_214 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_219 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_224 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (convert_element_type_147, primals_1, primals_5, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, slice_2, add, getitem_1, rsqrt, where, view, permute_1, permute_3, permute_5, getitem_2, getitem_3, getitem_4, getitem_5, add_3, getitem_7, rsqrt_1, view_12, addmm_4, view_14, add_7, getitem_9, rsqrt_2, view_16, permute_11, permute_13, permute_15, getitem_10, getitem_11, getitem_12, getitem_13, add_10, getitem_15, rsqrt_3, view_28, addmm_10, view_30, add_14, getitem_17, rsqrt_4, view_32, permute_21, permute_23, permute_25, getitem_18, getitem_19, getitem_20, getitem_21, add_17, getitem_23, rsqrt_5, view_44, addmm_16, view_46, add_21, getitem_25, rsqrt_6, view_48, permute_31, permute_33, permute_35, getitem_26, getitem_27, getitem_28, getitem_29, add_24, getitem_31, rsqrt_7, view_60, addmm_22, view_62, add_28, getitem_33, rsqrt_8, view_64, permute_41, permute_43, permute_45, getitem_34, getitem_35, getitem_36, getitem_37, add_31, getitem_39, rsqrt_9, view_76, addmm_28, view_78, add_35, getitem_41, rsqrt_10, view_80, permute_51, permute_53, permute_55, getitem_42, getitem_43, getitem_44, getitem_45, add_38, getitem_47, rsqrt_11, view_92, addmm_34, view_94, add_42, getitem_49, rsqrt_12, permute_60, permute_64, permute_68, permute_74, permute_79, permute_84, permute_88, permute_92, permute_96, permute_102, permute_107, permute_112, permute_116, permute_120, permute_124, permute_130, permute_135, permute_140, permute_144, permute_148, permute_152, permute_158, permute_163, permute_168, permute_172, permute_176, permute_180, permute_186, permute_191, permute_196, permute_200, permute_204, permute_208, permute_214, permute_219, permute_224)
        
def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 46881792, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (30522, 768), dtype=torch.float16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (512, 768), dtype=torch.float16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (768,), dtype=torch.float16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (768,), dtype=torch.float16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf6, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (768,), dtype=torch.float16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (768,), dtype=torch.float16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (768,), dtype=torch.float16, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf14, (768,), dtype=torch.float16, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf15, (768,), dtype=torch.float16, is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf16, (768,), dtype=torch.float16, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf17, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf18, (3072,), dtype=torch.float16, is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf19, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf20, (768,), dtype=torch.float16, is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf21, (768,), dtype=torch.float16, is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf22, (768,), dtype=torch.float16, is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf23, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf24, (768,), dtype=torch.float16, is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf25, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf26, (768,), dtype=torch.float16, is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf27, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf28, (768,), dtype=torch.float16, is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf29, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf30, (768,), dtype=torch.float16, is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf31, (768,), dtype=torch.float16, is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf32, (768,), dtype=torch.float16, is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf33, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf34, (3072,), dtype=torch.float16, is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf35, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf36, (768,), dtype=torch.float16, is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf37, (768,), dtype=torch.float16, is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf38, (768,), dtype=torch.float16, is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf39, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf40, (768,), dtype=torch.float16, is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf41, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf42, (768,), dtype=torch.float16, is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf43, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf44, (768,), dtype=torch.float16, is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf45, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf46, (768,), dtype=torch.float16, is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf47, (768,), dtype=torch.float16, is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf48, (768,), dtype=torch.float16, is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf49, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf50, (3072,), dtype=torch.float16, is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf51, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf52, (768,), dtype=torch.float16, is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf53, (768,), dtype=torch.float16, is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf54, (768,), dtype=torch.float16, is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf55, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf56, (768,), dtype=torch.float16, is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf57, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf58, (768,), dtype=torch.float16, is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf59, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf60, (768,), dtype=torch.float16, is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf61, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf62, (768,), dtype=torch.float16, is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf63, (768,), dtype=torch.float16, is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf64, (768,), dtype=torch.float16, is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf65, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf66, (3072,), dtype=torch.float16, is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf67, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf68, (768,), dtype=torch.float16, is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf69, (768,), dtype=torch.float16, is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf70, (768,), dtype=torch.float16, is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf71, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf72, (768,), dtype=torch.float16, is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf73, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf74, (768,), dtype=torch.float16, is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf75, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf76, (768,), dtype=torch.float16, is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf77, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf78, (768,), dtype=torch.float16, is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf79, (768,), dtype=torch.float16, is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf80, (768,), dtype=torch.float16, is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf81, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf82, (3072,), dtype=torch.float16, is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf83, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf84, (768,), dtype=torch.float16, is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf85, (768,), dtype=torch.float16, is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf86, (768,), dtype=torch.float16, is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf87, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf88, (768,), dtype=torch.float16, is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf89, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf90, (768,), dtype=torch.float16, is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf91, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf92, (768,), dtype=torch.float16, is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf93, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf94, (768,), dtype=torch.float16, is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf95, (768,), dtype=torch.float16, is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf96, (768,), dtype=torch.float16, is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf97, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf98, (3072,), dtype=torch.float16, is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf99, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf100, (768,), dtype=torch.float16, is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf101, (768,), dtype=torch.float16, is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf102, (768,), dtype=torch.float16, is_leaf=True)  # primals_103
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)