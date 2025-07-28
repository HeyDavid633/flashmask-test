import torch

class GraphModule(torch.nn.Module):
    
    def forward(self, L_self_modules_wte_parameters_weight_ : torch.nn.parameter.Parameter, s0 : torch.SymInt, L_input_ids_ : torch.Tensor, L_self_modules_wpe_parameters_weight_ : torch.nn.parameter.Parameter, L_attention_mask_ : torch.Tensor, L_batch_size_ : torch.SymInt, L_self_modules_h_modules_0_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_ln_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_ln_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_ln_2_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_ln_2_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_ln_f_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_ln_f_parameters_bias_ : torch.nn.parameter.Parameter, L_input_shape_1_ : torch.SymInt):
        l_self_modules_wte_parameters_weight_ = L_self_modules_wte_parameters_weight_
        l_input_ids_ = L_input_ids_
        l_self_modules_wpe_parameters_weight_ = L_self_modules_wpe_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_batch_size_ = L_batch_size_
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_0_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_0_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_0_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_0_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_0_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_1_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_1_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_1_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_1_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_1_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_2_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_2_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_2_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_2_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_2_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_3_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_3_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_3_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_3_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_3_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_4_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_4_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_4_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_4_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_4_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_5_modules_ln_1_parameters_weight_ = L_self_modules_h_modules_5_modules_ln_1_parameters_weight_
        l_self_modules_h_modules_5_modules_ln_1_parameters_bias_ = L_self_modules_h_modules_5_modules_ln_1_parameters_bias_
        l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_h_modules_5_modules_ln_2_parameters_weight_ = L_self_modules_h_modules_5_modules_ln_2_parameters_weight_
        l_self_modules_h_modules_5_modules_ln_2_parameters_bias_ = L_self_modules_h_modules_5_modules_ln_2_parameters_bias_
        l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_ln_f_parameters_weight_ = L_self_modules_ln_f_parameters_weight_
        l_self_modules_ln_f_parameters_bias_ = L_self_modules_ln_f_parameters_bias_
        l_input_shape_1_ = L_input_shape_1_
        inputs_embeds = torch.nn.functional.embedding(l_input_ids_, l_self_modules_wte_parameters_weight_, None, None, 2.0, False, False);  l_input_ids_ = l_self_modules_wte_parameters_weight_ = None
        cache_position = torch.arange(0, s0, device = device(type='cuda', index=0))
        position_ids = cache_position.unsqueeze(0)
        position_embeds = torch.nn.functional.embedding(position_ids, l_self_modules_wpe_parameters_weight_, None, None, 2.0, False, False);  position_ids = l_self_modules_wpe_parameters_weight_ = None
        to = position_embeds.to(device(type='cuda', index=0));  position_embeds = None
        hidden_states = inputs_embeds + to;  inputs_embeds = to = None
        attention_mask = l_attention_mask_.view(l_batch_size_, -1);  l_attention_mask_ = l_batch_size_ = None
        causal_mask = torch.full((s0, s0), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        causal_mask_1 = torch.triu(causal_mask, diagonal = 1);  causal_mask = None
        arange_1 = torch.arange(s0, device = device(type='cuda', index=0))
        reshape = cache_position.reshape(-1, 1);  cache_position = None
        gt = arange_1 > reshape;  arange_1 = reshape = None
        causal_mask_1 *= gt;  causal_mask_2 = causal_mask_1;  causal_mask_1 = gt = None
        getitem_19 = causal_mask_2[(None, None, slice(None, None, None), slice(None, None, None))];  causal_mask_2 = None
        causal_mask_3 = getitem_19.expand(1, 1, -1, -1);  getitem_19 = None
        causal_mask_4 = causal_mask_3.clone();  causal_mask_3 = None
        getitem_22 = causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        getitem_23 = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
        padding_mask = getitem_22 + getitem_23;  getitem_22 = getitem_23 = None
        padding_mask_1 = padding_mask == 0;  padding_mask = None
        getitem_24 = causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        masked_fill = getitem_24.masked_fill(padding_mask_1, -3.4028234663852886e+38);  getitem_24 = padding_mask_1 = None
        causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))] = masked_fill;  setitem = causal_mask_4;  masked_fill = setitem = None
        eq_1 = causal_mask_4 == -3.4028234663852886e+38
        all_1 = torch.all(eq_1, dim = -1, keepdim = True);  eq_1 = None
        invert = ~all_1;  all_1 = None
        causal_mask_5 = causal_mask_4.mul(invert);  causal_mask_4 = invert = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False);  hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(hidden_states_1, (768,), l_self_modules_h_modules_0_modules_ln_1_parameters_weight_, l_self_modules_h_modules_0_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = None
        view_1 = hidden_states_2.view(-1, 768);  hidden_states_2 = None
        x = torch.addmm(l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_, view_1, l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = view_1 = l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = None
        x_1 = x.view((1, s0, 2304));  x = None
        split = x_1.split(768, dim = 2);  x_1 = None
        query_states = split[0]
        key_states = split[1]
        value_states = split[2];  split = None
        view_3 = query_states.view((1, s0, -1, 64));  query_states = None
        query_states_1 = view_3.transpose(1, 2);  view_3 = None
        view_4 = key_states.view((1, s0, -1, 64));  key_states = None
        key_states_1 = view_4.transpose(1, 2);  view_4 = None
        view_5 = value_states.view((1, s0, -1, 64));  value_states = None
        value_states_1 = view_5.transpose(1, 2);  view_5 = None
        attention_mask_1 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        query = query_states_1.contiguous();  query_states_1 = None
        key = key_states_1.contiguous()
        value = value_states_1.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(query, key, value, attn_mask = attention_mask_1, dropout_p = 0.0, scale = None, is_causal = False);  query = key = value = attention_mask_1 = None
        transpose_3 = attn_output.transpose(1, 2);  attn_output = None
        attn_output_1 = transpose_3.contiguous();  transpose_3 = None
        reshape_1 = attn_output_1.reshape(1, s0, -1);  attn_output_1 = None
        attn_output_2 = reshape_1.contiguous();  reshape_1 = None
        view_6 = attn_output_2.view(-1, 768);  attn_output_2 = None
        x_2 = torch.addmm(l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_, view_6, l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = view_6 = l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = None
        x_3 = x_2.view((1, s0, 768));  x_2 = None
        attn_output_3 = torch.nn.functional.dropout(x_3, 0.1, False, False);  x_3 = None
        hidden_states_3 = attn_output_3 + hidden_states_1;  attn_output_3 = hidden_states_1 = None
        hidden_states_4 = torch.nn.functional.layer_norm(hidden_states_3, (768,), l_self_modules_h_modules_0_modules_ln_2_parameters_weight_, l_self_modules_h_modules_0_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_0_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_0_modules_ln_2_parameters_bias_ = None
        view_8 = hidden_states_4.view(-1, 768);  hidden_states_4 = None
        x_4 = torch.addmm(l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_, view_8, l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = view_8 = l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_5 = x_4.view((1, s0, 3072));  x_4 = None
        mul_1 = 0.5 * x_5
        pow_1 = torch.pow(x_5, 3.0)
        mul_2 = 0.044715 * pow_1;  pow_1 = None
        add_5 = x_5 + mul_2;  x_5 = mul_2 = None
        mul_3 = 0.7978845608028654 * add_5;  add_5 = None
        tanh = torch.tanh(mul_3);  mul_3 = None
        add_6 = 1.0 + tanh;  tanh = None
        hidden_states_5 = mul_1 * add_6;  mul_1 = add_6 = None
        view_10 = hidden_states_5.view(-1, 3072);  hidden_states_5 = None
        x_6 = torch.addmm(l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_, view_10, l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = view_10 = l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_7 = x_6.view((1, s0, 768));  x_6 = None
        hidden_states_6 = torch.nn.functional.dropout(x_7, 0.1, False, False);  x_7 = None
        hidden_states_7 = hidden_states_3 + hidden_states_6;  hidden_states_3 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(hidden_states_7, (768,), l_self_modules_h_modules_1_modules_ln_1_parameters_weight_, l_self_modules_h_modules_1_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = None
        view_12 = hidden_states_8.view(-1, 768);  hidden_states_8 = None
        x_8 = torch.addmm(l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_, view_12, l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = view_12 = l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = None
        x_9 = x_8.view((1, s0, 2304));  x_8 = None
        split_1 = x_9.split(768, dim = 2);  x_9 = None
        query_states_2 = split_1[0]
        key_states_2 = split_1[1]
        value_states_2 = split_1[2];  split_1 = None
        view_14 = query_states_2.view((1, s0, -1, 64));  query_states_2 = None
        query_states_3 = view_14.transpose(1, 2);  view_14 = None
        view_15 = key_states_2.view((1, s0, -1, 64));  key_states_2 = None
        key_states_3 = view_15.transpose(1, 2);  view_15 = None
        view_16 = value_states_2.view((1, s0, -1, 64));  value_states_2 = None
        value_states_3 = view_16.transpose(1, 2);  view_16 = None
        attention_mask_2 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        query_1 = query_states_3.contiguous();  query_states_3 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(query_1, key_1, value_1, attn_mask = attention_mask_2, dropout_p = 0.0, scale = None, is_causal = False);  query_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_7 = attn_output_4.transpose(1, 2);  attn_output_4 = None
        attn_output_5 = transpose_7.contiguous();  transpose_7 = None
        reshape_2 = attn_output_5.reshape(1, s0, -1);  attn_output_5 = None
        attn_output_6 = reshape_2.contiguous();  reshape_2 = None
        view_17 = attn_output_6.view(-1, 768);  attn_output_6 = None
        x_10 = torch.addmm(l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_, view_17, l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = view_17 = l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = None
        x_11 = x_10.view((1, s0, 768));  x_10 = None
        attn_output_7 = torch.nn.functional.dropout(x_11, 0.1, False, False);  x_11 = None
        hidden_states_9 = attn_output_7 + hidden_states_7;  attn_output_7 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(hidden_states_9, (768,), l_self_modules_h_modules_1_modules_ln_2_parameters_weight_, l_self_modules_h_modules_1_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_1_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_1_modules_ln_2_parameters_bias_ = None
        view_19 = hidden_states_10.view(-1, 768);  hidden_states_10 = None
        x_12 = torch.addmm(l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_, view_19, l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = view_19 = l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_13 = x_12.view((1, s0, 3072));  x_12 = None
        mul_5 = 0.5 * x_13
        pow_2 = torch.pow(x_13, 3.0)
        mul_6 = 0.044715 * pow_2;  pow_2 = None
        add_9 = x_13 + mul_6;  x_13 = mul_6 = None
        mul_7 = 0.7978845608028654 * add_9;  add_9 = None
        tanh_1 = torch.tanh(mul_7);  mul_7 = None
        add_10 = 1.0 + tanh_1;  tanh_1 = None
        hidden_states_11 = mul_5 * add_10;  mul_5 = add_10 = None
        view_21 = hidden_states_11.view(-1, 3072);  hidden_states_11 = None
        x_14 = torch.addmm(l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_, view_21, l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = view_21 = l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_15 = x_14.view((1, s0, 768));  x_14 = None
        hidden_states_12 = torch.nn.functional.dropout(x_15, 0.1, False, False);  x_15 = None
        hidden_states_13 = hidden_states_9 + hidden_states_12;  hidden_states_9 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(hidden_states_13, (768,), l_self_modules_h_modules_2_modules_ln_1_parameters_weight_, l_self_modules_h_modules_2_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = None
        view_23 = hidden_states_14.view(-1, 768);  hidden_states_14 = None
        x_16 = torch.addmm(l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_, view_23, l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = view_23 = l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = None
        x_17 = x_16.view((1, s0, 2304));  x_16 = None
        split_2 = x_17.split(768, dim = 2);  x_17 = None
        query_states_4 = split_2[0]
        key_states_4 = split_2[1]
        value_states_4 = split_2[2];  split_2 = None
        view_25 = query_states_4.view((1, s0, -1, 64));  query_states_4 = None
        query_states_5 = view_25.transpose(1, 2);  view_25 = None
        view_26 = key_states_4.view((1, s0, -1, 64));  key_states_4 = None
        key_states_5 = view_26.transpose(1, 2);  view_26 = None
        view_27 = value_states_4.view((1, s0, -1, 64));  value_states_4 = None
        value_states_5 = view_27.transpose(1, 2);  view_27 = None
        attention_mask_3 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        query_2 = query_states_5.contiguous();  query_states_5 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(query_2, key_2, value_2, attn_mask = attention_mask_3, dropout_p = 0.0, scale = None, is_causal = False);  query_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_11 = attn_output_8.transpose(1, 2);  attn_output_8 = None
        attn_output_9 = transpose_11.contiguous();  transpose_11 = None
        reshape_3 = attn_output_9.reshape(1, s0, -1);  attn_output_9 = None
        attn_output_10 = reshape_3.contiguous();  reshape_3 = None
        view_28 = attn_output_10.view(-1, 768);  attn_output_10 = None
        x_18 = torch.addmm(l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_, view_28, l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = view_28 = l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = None
        x_19 = x_18.view((1, s0, 768));  x_18 = None
        attn_output_11 = torch.nn.functional.dropout(x_19, 0.1, False, False);  x_19 = None
        hidden_states_15 = attn_output_11 + hidden_states_13;  attn_output_11 = hidden_states_13 = None
        hidden_states_16 = torch.nn.functional.layer_norm(hidden_states_15, (768,), l_self_modules_h_modules_2_modules_ln_2_parameters_weight_, l_self_modules_h_modules_2_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_2_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_2_modules_ln_2_parameters_bias_ = None
        view_30 = hidden_states_16.view(-1, 768);  hidden_states_16 = None
        x_20 = torch.addmm(l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_, view_30, l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = view_30 = l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_21 = x_20.view((1, s0, 3072));  x_20 = None
        mul_9 = 0.5 * x_21
        pow_3 = torch.pow(x_21, 3.0)
        mul_10 = 0.044715 * pow_3;  pow_3 = None
        add_13 = x_21 + mul_10;  x_21 = mul_10 = None
        mul_11 = 0.7978845608028654 * add_13;  add_13 = None
        tanh_2 = torch.tanh(mul_11);  mul_11 = None
        add_14 = 1.0 + tanh_2;  tanh_2 = None
        hidden_states_17 = mul_9 * add_14;  mul_9 = add_14 = None
        view_32 = hidden_states_17.view(-1, 3072);  hidden_states_17 = None
        x_22 = torch.addmm(l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_, view_32, l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = view_32 = l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_23 = x_22.view((1, s0, 768));  x_22 = None
        hidden_states_18 = torch.nn.functional.dropout(x_23, 0.1, False, False);  x_23 = None
        hidden_states_19 = hidden_states_15 + hidden_states_18;  hidden_states_15 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(hidden_states_19, (768,), l_self_modules_h_modules_3_modules_ln_1_parameters_weight_, l_self_modules_h_modules_3_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = None
        view_34 = hidden_states_20.view(-1, 768);  hidden_states_20 = None
        x_24 = torch.addmm(l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_, view_34, l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = view_34 = l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = None
        x_25 = x_24.view((1, s0, 2304));  x_24 = None
        split_3 = x_25.split(768, dim = 2);  x_25 = None
        query_states_6 = split_3[0]
        key_states_6 = split_3[1]
        value_states_6 = split_3[2];  split_3 = None
        view_36 = query_states_6.view((1, s0, -1, 64));  query_states_6 = None
        query_states_7 = view_36.transpose(1, 2);  view_36 = None
        view_37 = key_states_6.view((1, s0, -1, 64));  key_states_6 = None
        key_states_7 = view_37.transpose(1, 2);  view_37 = None
        view_38 = value_states_6.view((1, s0, -1, 64));  value_states_6 = None
        value_states_7 = view_38.transpose(1, 2);  view_38 = None
        attention_mask_4 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        query_3 = query_states_7.contiguous();  query_states_7 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(query_3, key_3, value_3, attn_mask = attention_mask_4, dropout_p = 0.0, scale = None, is_causal = False);  query_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_15 = attn_output_12.transpose(1, 2);  attn_output_12 = None
        attn_output_13 = transpose_15.contiguous();  transpose_15 = None
        reshape_4 = attn_output_13.reshape(1, s0, -1);  attn_output_13 = None
        attn_output_14 = reshape_4.contiguous();  reshape_4 = None
        view_39 = attn_output_14.view(-1, 768);  attn_output_14 = None
        x_26 = torch.addmm(l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_, view_39, l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = view_39 = l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = None
        x_27 = x_26.view((1, s0, 768));  x_26 = None
        attn_output_15 = torch.nn.functional.dropout(x_27, 0.1, False, False);  x_27 = None
        hidden_states_21 = attn_output_15 + hidden_states_19;  attn_output_15 = hidden_states_19 = None
        hidden_states_22 = torch.nn.functional.layer_norm(hidden_states_21, (768,), l_self_modules_h_modules_3_modules_ln_2_parameters_weight_, l_self_modules_h_modules_3_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_3_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_3_modules_ln_2_parameters_bias_ = None
        view_41 = hidden_states_22.view(-1, 768);  hidden_states_22 = None
        x_28 = torch.addmm(l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_, view_41, l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = view_41 = l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_29 = x_28.view((1, s0, 3072));  x_28 = None
        mul_13 = 0.5 * x_29
        pow_4 = torch.pow(x_29, 3.0)
        mul_14 = 0.044715 * pow_4;  pow_4 = None
        add_17 = x_29 + mul_14;  x_29 = mul_14 = None
        mul_15 = 0.7978845608028654 * add_17;  add_17 = None
        tanh_3 = torch.tanh(mul_15);  mul_15 = None
        add_18 = 1.0 + tanh_3;  tanh_3 = None
        hidden_states_23 = mul_13 * add_18;  mul_13 = add_18 = None
        view_43 = hidden_states_23.view(-1, 3072);  hidden_states_23 = None
        x_30 = torch.addmm(l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_, view_43, l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = view_43 = l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_31 = x_30.view((1, s0, 768));  x_30 = None
        hidden_states_24 = torch.nn.functional.dropout(x_31, 0.1, False, False);  x_31 = None
        hidden_states_25 = hidden_states_21 + hidden_states_24;  hidden_states_21 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(hidden_states_25, (768,), l_self_modules_h_modules_4_modules_ln_1_parameters_weight_, l_self_modules_h_modules_4_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = None
        view_45 = hidden_states_26.view(-1, 768);  hidden_states_26 = None
        x_32 = torch.addmm(l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_, view_45, l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = view_45 = l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = None
        x_33 = x_32.view((1, s0, 2304));  x_32 = None
        split_4 = x_33.split(768, dim = 2);  x_33 = None
        query_states_8 = split_4[0]
        key_states_8 = split_4[1]
        value_states_8 = split_4[2];  split_4 = None
        view_47 = query_states_8.view((1, s0, -1, 64));  query_states_8 = None
        query_states_9 = view_47.transpose(1, 2);  view_47 = None
        view_48 = key_states_8.view((1, s0, -1, 64));  key_states_8 = None
        key_states_9 = view_48.transpose(1, 2);  view_48 = None
        view_49 = value_states_8.view((1, s0, -1, 64));  value_states_8 = None
        value_states_9 = view_49.transpose(1, 2);  view_49 = None
        attention_mask_5 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))]
        query_4 = query_states_9.contiguous();  query_states_9 = None
        key_4 = key_states_9.contiguous()
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(query_4, key_4, value_4, attn_mask = attention_mask_5, dropout_p = 0.0, scale = None, is_causal = False);  query_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_19 = attn_output_16.transpose(1, 2);  attn_output_16 = None
        attn_output_17 = transpose_19.contiguous();  transpose_19 = None
        reshape_5 = attn_output_17.reshape(1, s0, -1);  attn_output_17 = None
        attn_output_18 = reshape_5.contiguous();  reshape_5 = None
        view_50 = attn_output_18.view(-1, 768);  attn_output_18 = None
        x_34 = torch.addmm(l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_, view_50, l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = view_50 = l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = None
        x_35 = x_34.view((1, s0, 768));  x_34 = None
        attn_output_19 = torch.nn.functional.dropout(x_35, 0.1, False, False);  x_35 = None
        hidden_states_27 = attn_output_19 + hidden_states_25;  attn_output_19 = hidden_states_25 = None
        hidden_states_28 = torch.nn.functional.layer_norm(hidden_states_27, (768,), l_self_modules_h_modules_4_modules_ln_2_parameters_weight_, l_self_modules_h_modules_4_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_4_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_4_modules_ln_2_parameters_bias_ = None
        view_52 = hidden_states_28.view(-1, 768);  hidden_states_28 = None
        x_36 = torch.addmm(l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_, view_52, l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = view_52 = l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_37 = x_36.view((1, s0, 3072));  x_36 = None
        mul_17 = 0.5 * x_37
        pow_5 = torch.pow(x_37, 3.0)
        mul_18 = 0.044715 * pow_5;  pow_5 = None
        add_21 = x_37 + mul_18;  x_37 = mul_18 = None
        mul_19 = 0.7978845608028654 * add_21;  add_21 = None
        tanh_4 = torch.tanh(mul_19);  mul_19 = None
        add_22 = 1.0 + tanh_4;  tanh_4 = None
        hidden_states_29 = mul_17 * add_22;  mul_17 = add_22 = None
        view_54 = hidden_states_29.view(-1, 3072);  hidden_states_29 = None
        x_38 = torch.addmm(l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_, view_54, l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = view_54 = l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_39 = x_38.view((1, s0, 768));  x_38 = None
        hidden_states_30 = torch.nn.functional.dropout(x_39, 0.1, False, False);  x_39 = None
        hidden_states_31 = hidden_states_27 + hidden_states_30;  hidden_states_27 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(hidden_states_31, (768,), l_self_modules_h_modules_5_modules_ln_1_parameters_weight_, l_self_modules_h_modules_5_modules_ln_1_parameters_bias_, 1e-05);  l_self_modules_h_modules_5_modules_ln_1_parameters_weight_ = l_self_modules_h_modules_5_modules_ln_1_parameters_bias_ = None
        view_56 = hidden_states_32.view(-1, 768);  hidden_states_32 = None
        x_40 = torch.addmm(l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_, view_56, l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_);  l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = view_56 = l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = None
        x_41 = x_40.view((1, s0, 2304));  x_40 = None
        split_5 = x_41.split(768, dim = 2);  x_41 = None
        query_states_10 = split_5[0]
        key_states_10 = split_5[1]
        value_states_10 = split_5[2];  split_5 = None
        view_58 = query_states_10.view((1, s0, -1, 64));  query_states_10 = None
        query_states_11 = view_58.transpose(1, 2);  view_58 = None
        view_59 = key_states_10.view((1, s0, -1, 64));  key_states_10 = None
        key_states_11 = view_59.transpose(1, 2);  view_59 = None
        view_60 = value_states_10.view((1, s0, -1, 64));  value_states_10 = None
        value_states_11 = view_60.transpose(1, 2);  view_60 = None
        attention_mask_6 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, s0, None))];  causal_mask_5 = None
        query_5 = query_states_11.contiguous();  query_states_11 = None
        key_5 = key_states_11.contiguous()
        value_5 = value_states_11.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(query_5, key_5, value_5, attn_mask = attention_mask_6, dropout_p = 0.0, scale = None, is_causal = False);  query_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_23 = attn_output_20.transpose(1, 2);  attn_output_20 = None
        attn_output_21 = transpose_23.contiguous();  transpose_23 = None
        reshape_6 = attn_output_21.reshape(1, s0, -1);  attn_output_21 = None
        attn_output_22 = reshape_6.contiguous();  reshape_6 = None
        view_61 = attn_output_22.view(-1, 768);  attn_output_22 = None
        x_42 = torch.addmm(l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_, view_61, l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = view_61 = l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = None
        x_43 = x_42.view((1, s0, 768));  x_42 = None
        attn_output_23 = torch.nn.functional.dropout(x_43, 0.1, False, False);  x_43 = None
        hidden_states_33 = attn_output_23 + hidden_states_31;  attn_output_23 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(hidden_states_33, (768,), l_self_modules_h_modules_5_modules_ln_2_parameters_weight_, l_self_modules_h_modules_5_modules_ln_2_parameters_bias_, 1e-05);  l_self_modules_h_modules_5_modules_ln_2_parameters_weight_ = l_self_modules_h_modules_5_modules_ln_2_parameters_bias_ = None
        view_63 = hidden_states_34.view(-1, 768);  hidden_states_34 = None
        x_44 = torch.addmm(l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_, view_63, l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_);  l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = view_63 = l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = None
        x_45 = x_44.view((1, s0, 3072));  x_44 = None
        mul_21 = 0.5 * x_45
        pow_6 = torch.pow(x_45, 3.0)
        mul_22 = 0.044715 * pow_6;  pow_6 = None
        add_25 = x_45 + mul_22;  x_45 = mul_22 = None
        mul_23 = 0.7978845608028654 * add_25;  add_25 = None
        tanh_5 = torch.tanh(mul_23);  mul_23 = None
        add_26 = 1.0 + tanh_5;  tanh_5 = None
        hidden_states_35 = mul_21 * add_26;  mul_21 = add_26 = None
        view_65 = hidden_states_35.view(-1, 3072);  hidden_states_35 = None
        x_46 = torch.addmm(l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_, view_65, l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_);  l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = view_65 = l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = None
        x_47 = x_46.view((1, s0, 768));  x_46 = s0 = None
        hidden_states_36 = torch.nn.functional.dropout(x_47, 0.1, False, False);  x_47 = None
        hidden_states_37 = hidden_states_33 + hidden_states_36;  hidden_states_33 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(hidden_states_37, (768,), l_self_modules_ln_f_parameters_weight_, l_self_modules_ln_f_parameters_bias_, 1e-05);  hidden_states_37 = l_self_modules_ln_f_parameters_weight_ = l_self_modules_ln_f_parameters_bias_ = None
        hidden_states_39 = hidden_states_38.view((-1, l_input_shape_1_, 768));  hidden_states_38 = l_input_shape_1_ = None
        return (value_states_1, key_states_1, value_states_3, key_states_3, value_states_5, key_states_5, value_states_7, key_states_7, value_states_9, key_states_9, value_states_11, key_states_11, hidden_states_39)
        