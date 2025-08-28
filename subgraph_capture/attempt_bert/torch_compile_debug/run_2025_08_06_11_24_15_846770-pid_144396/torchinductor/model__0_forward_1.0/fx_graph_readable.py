class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128]", primals_2: "f16[30522, 768]", primals_3: "i64[1, 512]", primals_4: "f16[512, 768]", primals_5: "f16[768]", primals_6: "f16[768]", primals_7: "i64[1, 128]", primals_8: "f16[768, 768]", primals_9: "f16[768]", primals_10: "f16[768, 768]", primals_11: "f16[768]", primals_12: "f16[768, 768]", primals_13: "f16[768]", primals_14: "f16[768, 768]", primals_15: "f16[768]", primals_16: "f16[768]", primals_17: "f16[768]", primals_18: "f16[3072, 768]", primals_19: "f16[3072]", primals_20: "f16[768, 3072]", primals_21: "f16[768]", primals_22: "f16[768]", primals_23: "f16[768]", primals_24: "f16[768, 768]", primals_25: "f16[768]", primals_26: "f16[768, 768]", primals_27: "f16[768]", primals_28: "f16[768, 768]", primals_29: "f16[768]", primals_30: "f16[768, 768]", primals_31: "f16[768]", primals_32: "f16[768]", primals_33: "f16[768]", primals_34: "f16[3072, 768]", primals_35: "f16[3072]", primals_36: "f16[768, 3072]", primals_37: "f16[768]", primals_38: "f16[768]", primals_39: "f16[768]", primals_40: "f16[768, 768]", primals_41: "f16[768]", primals_42: "f16[768, 768]", primals_43: "f16[768]", primals_44: "f16[768, 768]", primals_45: "f16[768]", primals_46: "f16[768, 768]", primals_47: "f16[768]", primals_48: "f16[768]", primals_49: "f16[768]", primals_50: "f16[3072, 768]", primals_51: "f16[3072]", primals_52: "f16[768, 3072]", primals_53: "f16[768]", primals_54: "f16[768]", primals_55: "f16[768]", primals_56: "f16[768, 768]", primals_57: "f16[768]", primals_58: "f16[768, 768]", primals_59: "f16[768]", primals_60: "f16[768, 768]", primals_61: "f16[768]", primals_62: "f16[768, 768]", primals_63: "f16[768]", primals_64: "f16[768]", primals_65: "f16[768]", primals_66: "f16[3072, 768]", primals_67: "f16[3072]", primals_68: "f16[768, 3072]", primals_69: "f16[768]", primals_70: "f16[768]", primals_71: "f16[768]", primals_72: "f16[768, 768]", primals_73: "f16[768]", primals_74: "f16[768, 768]", primals_75: "f16[768]", primals_76: "f16[768, 768]", primals_77: "f16[768]", primals_78: "f16[768, 768]", primals_79: "f16[768]", primals_80: "f16[768]", primals_81: "f16[768]", primals_82: "f16[3072, 768]", primals_83: "f16[3072]", primals_84: "f16[768, 3072]", primals_85: "f16[768]", primals_86: "f16[768]", primals_87: "f16[768]", primals_88: "f16[768, 768]", primals_89: "f16[768]", primals_90: "f16[768, 768]", primals_91: "f16[768]", primals_92: "f16[768, 768]", primals_93: "f16[768]", primals_94: "f16[768, 768]", primals_95: "f16[768]", primals_96: "f16[768]", primals_97: "f16[768]", primals_98: "f16[3072, 768]", primals_99: "f16[3072]", primals_100: "f16[768, 3072]", primals_101: "f16[768]", primals_102: "f16[768]", primals_103: "f16[768]"):
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:111 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f16[1, 128, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:124 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f16[1, 128, 768]" = torch.ops.aten.embedding.default(primals_4, slice_2);  primals_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:126 in forward, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:127 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        convert_element_type: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        convert_element_type_1: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(primals_7, 1);  primals_7 = None
        unsqueeze_1: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand: "i64[1, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [1, 1, 128, 128]);  unsqueeze_1 = None
        convert_element_type_2: "f16[1, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(expand, torch.float16);  expand = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        full_default: "f16[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1: "f16[1, 1, 128, 128]" = torch.ops.aten.sub.Tensor(full_default, convert_element_type_2);  full_default = convert_element_type_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_3: "b8[1, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 128, 128]" = torch.ops.aten.where.self(convert_element_type_3, full_default_1, sub_1);  convert_element_type_3 = full_default_1 = sub_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_1, [128, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_9, view, permute);  primals_9 = None
        view_1: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm, [1, 128, 768]);  addmm = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_2: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_1: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_11, view, permute_2);  primals_11 = None
        view_4: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_5: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_4: "f16[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm_2: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_13, view, permute_4);  primals_13 = None
        view_7: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_2, [1, 128, 768]);  addmm_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_8: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_1: "f16[1, 12, 128, 128]" = torch.ops.aten.expand.default(where, [1, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_1, True)
        getitem_2: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_3: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention[1]
        getitem_4: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_5: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_6: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])
        view_9: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_6, [1, -1, 768]);  permute_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_10: "f16[128, 768]" = torch.ops.aten.view.default(view_9, [128, 768]);  view_9 = None
        permute_7: "f16[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        addmm_3: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_15, view_10, permute_7);  primals_15 = view_10 = None
        view_11: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_3: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_11, convert_element_type_1);  view_11 = convert_element_type_1 = None
        convert_element_type_16: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_7);  convert_element_type_16 = None
        mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_16);  mul_2 = None
        add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_17);  mul_3 = primals_17 = None
        convert_element_type_17: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_12: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_17, [128, 768])
        permute_8: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        addmm_4: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_19, view_12, permute_8);  primals_19 = None
        view_13: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_21: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        convert_element_type_22: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_14: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_22, [128, 3072]);  convert_element_type_22 = None
        permute_9: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        addmm_5: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_21, view_14, permute_9);  primals_21 = None
        view_15: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_7: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_15, convert_element_type_17);  view_15 = convert_element_type_17 = None
        convert_element_type_26: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_9);  convert_element_type_26 = None
        mul_7: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_22);  mul_7 = None
        add_9: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_23);  mul_8 = primals_23 = None
        convert_element_type_27: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_16: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_27, [128, 768])
        permute_10: "f16[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_6: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_25, view_16, permute_10);  primals_25 = None
        view_17: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_6, [1, 128, 768]);  addmm_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_18: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_17, [1, -1, 12, 64]);  view_17 = None
        permute_11: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_12: "f16[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_7: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_27, view_16, permute_12);  primals_27 = None
        view_20: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_21: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_20, [1, -1, 12, 64]);  view_20 = None
        permute_13: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_14: "f16[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        addmm_8: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_29, view_16, permute_14);  primals_29 = None
        view_23: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_8, [1, 128, 768]);  addmm_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_24: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_23, [1, -1, 12, 64]);  view_23 = None
        permute_15: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_1, True)
        getitem_10: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_11: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_12: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_13: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_16: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3])
        view_25: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_16, [1, -1, 768]);  permute_16 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_26: "f16[128, 768]" = torch.ops.aten.view.default(view_25, [128, 768]);  view_25 = None
        permute_17: "f16[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        addmm_9: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_31, view_26, permute_17);  primals_31 = view_26 = None
        view_27: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_10: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_27, convert_element_type_27);  view_27 = convert_element_type_27 = None
        convert_element_type_40: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_15: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_15);  convert_element_type_40 = None
        mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_32);  mul_9 = None
        add_12: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_33);  mul_10 = primals_33 = None
        convert_element_type_41: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_28: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_41, [128, 768])
        permute_18: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        addmm_10: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_35, view_28, permute_18);  primals_35 = None
        view_29: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_45: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_11: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.5)
        mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.7071067811865476);  convert_element_type_45 = None
        erf_1: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        convert_element_type_46: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_30: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_46, [128, 3072]);  convert_element_type_46 = None
        permute_19: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_11: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_37, view_30, permute_19);  primals_37 = None
        view_31: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_14: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_31, convert_element_type_41);  view_31 = convert_element_type_41 = None
        convert_element_type_50: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_17);  convert_element_type_50 = None
        mul_14: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_38);  mul_14 = None
        add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_39);  mul_15 = primals_39 = None
        convert_element_type_51: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_32: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_51, [128, 768])
        permute_20: "f16[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        addmm_12: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_41, view_32, permute_20);  primals_41 = None
        view_33: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_12, [1, 128, 768]);  addmm_12 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_34: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_33, [1, -1, 12, 64]);  view_33 = None
        permute_21: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_22: "f16[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        addmm_13: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_43, view_32, permute_22);  primals_43 = None
        view_36: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_37: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_36, [1, -1, 12, 64]);  view_36 = None
        permute_23: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_24: "f16[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        addmm_14: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_45, view_32, permute_24);  primals_45 = None
        view_39: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_14, [1, 128, 768]);  addmm_14 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_40: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_39, [1, -1, 12, 64]);  view_39 = None
        permute_25: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_1, True)
        getitem_18: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_19: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_20: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_21: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_26: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])
        view_41: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_26, [1, -1, 768]);  permute_26 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_42: "f16[128, 768]" = torch.ops.aten.view.default(view_41, [128, 768]);  view_41 = None
        permute_27: "f16[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        addmm_15: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_47, view_42, permute_27);  primals_47 = view_42 = None
        view_43: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_17: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_43, convert_element_type_51);  view_43 = convert_element_type_51 = None
        convert_element_type_64: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_23: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_23);  convert_element_type_64 = None
        mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
        mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_48);  mul_16 = None
        add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_49);  mul_17 = primals_49 = None
        convert_element_type_65: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_44: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_65, [128, 768])
        permute_28: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        addmm_16: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_51, view_44, permute_28);  primals_51 = None
        view_45: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_69: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_18: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.5)
        mul_19: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.7071067811865476);  convert_element_type_69 = None
        erf_2: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        convert_element_type_70: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_46: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_70, [128, 3072]);  convert_element_type_70 = None
        permute_29: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        addmm_17: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_53, view_46, permute_29);  primals_53 = None
        view_47: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_21: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_47, convert_element_type_65);  view_47 = convert_element_type_65 = None
        convert_element_type_74: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_74, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_7: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_74, getitem_25);  convert_element_type_74 = None
        mul_21: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
        mul_22: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_54);  mul_21 = None
        add_23: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_55);  mul_22 = primals_55 = None
        convert_element_type_75: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_48: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_75, [128, 768])
        permute_30: "f16[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_18: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_57, view_48, permute_30);  primals_57 = None
        view_49: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_18, [1, 128, 768]);  addmm_18 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_50: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_49, [1, -1, 12, 64]);  view_49 = None
        permute_31: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_32: "f16[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_19: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_59, view_48, permute_32);  primals_59 = None
        view_52: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_53: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_52, [1, -1, 12, 64]);  view_52 = None
        permute_33: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_34: "f16[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_20: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_61, view_48, permute_34);  primals_61 = None
        view_55: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_20, [1, 128, 768]);  addmm_20 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_56: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_55, [1, -1, 12, 64]);  view_55 = None
        permute_35: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_1, True)
        getitem_26: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_27: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_28: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_36: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3])
        view_57: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_36, [1, -1, 768]);  permute_36 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_58: "f16[128, 768]" = torch.ops.aten.view.default(view_57, [128, 768]);  view_57 = None
        permute_37: "f16[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        addmm_21: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_63, view_58, permute_37);  primals_63 = view_58 = None
        view_59: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_24: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_59, convert_element_type_75);  view_59 = convert_element_type_75 = None
        convert_element_type_88: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_31: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_31);  convert_element_type_88 = None
        mul_23: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
        mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_64);  mul_23 = None
        add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_65);  mul_24 = primals_65 = None
        convert_element_type_89: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_60: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_89, [128, 768])
        permute_38: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        addmm_22: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_67, view_60, permute_38);  primals_67 = None
        view_61: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_93: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_25: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.5)
        mul_26: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.7071067811865476);  convert_element_type_93 = None
        erf_3: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        convert_element_type_94: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_62: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_94, [128, 3072]);  convert_element_type_94 = None
        permute_39: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        addmm_23: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_69, view_62, permute_39);  primals_69 = None
        view_63: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_28: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_63, convert_element_type_89);  view_63 = convert_element_type_89 = None
        convert_element_type_98: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_33);  convert_element_type_98 = None
        mul_28: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
        mul_29: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_70);  mul_28 = None
        add_30: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_71);  mul_29 = primals_71 = None
        convert_element_type_99: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_64: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_99, [128, 768])
        permute_40: "f16[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_24: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_73, view_64, permute_40);  primals_73 = None
        view_65: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_24, [1, 128, 768]);  addmm_24 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_66: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_65, [1, -1, 12, 64]);  view_65 = None
        permute_41: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_42: "f16[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        addmm_25: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_75, view_64, permute_42);  primals_75 = None
        view_68: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_25, [1, 128, 768]);  addmm_25 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_69: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_68, [1, -1, 12, 64]);  view_68 = None
        permute_43: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_44: "f16[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        addmm_26: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_77, view_64, permute_44);  primals_77 = None
        view_71: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_26, [1, 128, 768]);  addmm_26 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_72: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_71, [1, -1, 12, 64]);  view_71 = None
        permute_45: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_1, True)
        getitem_34: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_35: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_36: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_37: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_46: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3])
        view_73: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_46, [1, -1, 768]);  permute_46 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_74: "f16[128, 768]" = torch.ops.aten.view.default(view_73, [128, 768]);  view_73 = None
        permute_47: "f16[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        addmm_27: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_79, view_74, permute_47);  primals_79 = view_74 = None
        view_75: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_27, [1, 128, 768]);  addmm_27 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_31: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_75, convert_element_type_99);  view_75 = convert_element_type_99 = None
        convert_element_type_112: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_39: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_10: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_39);  convert_element_type_112 = None
        mul_30: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        mul_31: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_80);  mul_30 = None
        add_33: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_81);  mul_31 = primals_81 = None
        convert_element_type_113: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_76: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_113, [128, 768])
        permute_48: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        addmm_28: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_83, view_76, permute_48);  primals_83 = None
        view_77: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_117: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_32: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.5)
        mul_33: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.7071067811865476);  convert_element_type_117 = None
        erf_4: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        convert_element_type_118: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_78: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_118, [128, 3072]);  convert_element_type_118 = None
        permute_49: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        addmm_29: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_85, view_78, permute_49);  primals_85 = None
        view_79: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_29, [1, 128, 768]);  addmm_29 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_35: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_79, convert_element_type_113);  view_79 = convert_element_type_113 = None
        convert_element_type_122: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_122, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_122, getitem_41);  convert_element_type_122 = None
        mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = None
        mul_36: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_86);  mul_35 = None
        add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_87);  mul_36 = primals_87 = None
        convert_element_type_123: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_80: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_123, [128, 768])
        permute_50: "f16[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        addmm_30: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_89, view_80, permute_50);  primals_89 = None
        view_81: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_30, [1, 128, 768]);  addmm_30 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_82: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_81, [1, -1, 12, 64]);  view_81 = None
        permute_51: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_52: "f16[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        addmm_31: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_91, view_80, permute_52);  primals_91 = None
        view_84: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_31, [1, 128, 768]);  addmm_31 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_85: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_84, [1, -1, 12, 64]);  view_84 = None
        permute_53: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_54: "f16[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        addmm_32: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_93, view_80, permute_54);  primals_93 = None
        view_87: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_32, [1, 128, 768]);  addmm_32 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:384 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_88: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(view_87, [1, -1, 12, 64]);  view_87 = None
        permute_55: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_1, True);  expand_1 = None
        getitem_42: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_43: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_44: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_45: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:388 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_56: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3])
        view_89: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_56, [1, -1, 768]);  permute_56 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        view_90: "f16[128, 768]" = torch.ops.aten.view.default(view_89, [128, 768]);  view_89 = None
        permute_57: "f16[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        addmm_33: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_95, view_90, permute_57);  primals_95 = view_90 = None
        view_91: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_33, [1, 128, 768]);  addmm_33 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:491 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_38: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_91, convert_element_type_123);  view_91 = convert_element_type_123 = None
        convert_element_type_136: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_47: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_47);  convert_element_type_136 = None
        mul_37: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
        mul_38: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_96);  mul_37 = None
        add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_97);  mul_38 = primals_97 = None
        convert_element_type_137: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        view_92: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_137, [128, 768])
        permute_58: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        addmm_34: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_99, view_92, permute_58);  primals_99 = None
        view_93: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/activations.py:69 in forward, code: return self.act(input)
        convert_element_type_141: "f32[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_39: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.5)
        mul_40: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.7071067811865476);  convert_element_type_141 = None
        erf_5: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        convert_element_type_142: "f16[1, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        view_94: "f16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_142, [128, 3072]);  convert_element_type_142 = None
        permute_59: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        addmm_35: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_101, view_94, permute_59);  primals_101 = None
        view_95: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_35, [1, 128, 768]);  addmm_35 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:495 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_42: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_95, convert_element_type_137);  view_95 = convert_element_type_137 = None
        convert_element_type_146: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_146, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_13: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_146, getitem_49);  convert_element_type_146 = None
        mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
        mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_102);  mul_42 = None
        add_44: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_103);  mul_43 = primals_103 = None
        convert_element_type_147: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_60: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_64: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_68: "f16[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_74: "f16[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_79: "f16[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_84: "f16[768, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_88: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_92: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_96: "f16[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_102: "f16[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_107: "f16[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_112: "f16[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_116: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_120: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_124: "f16[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_130: "f16[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_135: "f16[768, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_140: "f16[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_144: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_148: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_152: "f16[768, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_158: "f16[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_163: "f16[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_168: "f16[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_172: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_176: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_180: "f16[768, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_186: "f16[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_191: "f16[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_196: "f16[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:433 in ff_chunk, code: x = self.lin2(x)
        permute_200: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:431 in ff_chunk, code: x = self.lin1(input)
        permute_204: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:412 in forward, code: attn_output = self.out_lin(attn_output)
        permute_208: "f16[768, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        permute_214: "f16[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        permute_219: "f16[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py:390 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        permute_224: "f16[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (convert_element_type_147, primals_1, primals_5, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, slice_2, add, getitem_1, rsqrt, where, view, permute_1, permute_3, permute_5, getitem_2, getitem_3, getitem_4, getitem_5, add_3, getitem_7, rsqrt_1, view_12, addmm_4, view_14, add_7, getitem_9, rsqrt_2, view_16, permute_11, permute_13, permute_15, getitem_10, getitem_11, getitem_12, getitem_13, add_10, getitem_15, rsqrt_3, view_28, addmm_10, view_30, add_14, getitem_17, rsqrt_4, view_32, permute_21, permute_23, permute_25, getitem_18, getitem_19, getitem_20, getitem_21, add_17, getitem_23, rsqrt_5, view_44, addmm_16, view_46, add_21, getitem_25, rsqrt_6, view_48, permute_31, permute_33, permute_35, getitem_26, getitem_27, getitem_28, getitem_29, add_24, getitem_31, rsqrt_7, view_60, addmm_22, view_62, add_28, getitem_33, rsqrt_8, view_64, permute_41, permute_43, permute_45, getitem_34, getitem_35, getitem_36, getitem_37, add_31, getitem_39, rsqrt_9, view_76, addmm_28, view_78, add_35, getitem_41, rsqrt_10, view_80, permute_51, permute_53, permute_55, getitem_42, getitem_43, getitem_44, getitem_45, add_38, getitem_47, rsqrt_11, view_92, addmm_34, view_94, add_42, getitem_49, rsqrt_12, permute_60, permute_64, permute_68, permute_74, permute_79, permute_84, permute_88, permute_92, permute_96, permute_102, permute_107, permute_112, permute_116, permute_120, permute_124, permute_130, permute_135, permute_140, permute_144, permute_148, permute_152, permute_158, permute_163, permute_168, permute_172, permute_176, permute_180, permute_186, permute_191, permute_196, permute_200, permute_204, permute_208, permute_214, permute_219, permute_224)
        