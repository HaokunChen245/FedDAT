import torch
import torch.nn as nn
import torch.nn.functional as F

def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, names, device, model_dim=768, adapter_reduction_factor=16):
        super().__init__()
        self.actv = nn.ReLU()
        self.scaling = 1.0
        self.gating = False
        print(names)

        if isinstance(names, str):
            names = [names]
        self.adapter_dict = {}
        for name in names:
            if 'adapter' in name:
                n = f'{name}_down'
                setattr(self, n, nn.Linear(model_dim, model_dim//adapter_reduction_factor).to(device))
                m = getattr(self, n)
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True
                n = f'{name}_up'
                setattr(self, n, nn.Linear(model_dim//adapter_reduction_factor, model_dim).to(device))
                m = getattr(self, n)
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True

            elif name in ['gating']:
                # for each client we init a spec gating
                setattr(self, f'{name}_module', nn.Linear(model_dim, 2).to(device))
                m = getattr(self, f'{name}_module')
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True

        if hasattr(self, 'adapter_2_down'):
            for m in [self.adapter_2_down, self.adapter_2_up]:
                for p in m.parameters():
                    p.requires_grad = False

    def deactivate_gating(self):
        self.gating = False

    def activate_gating(self):
        self.gating = True

    def set_active_adapter(self, name):
        if isinstance(name, str):
            self.active_adapter_down = getattr(self, f'{name}_down')
            self.active_adapter_up = getattr(self, f'{name}_up')

        if name == 'adapter_0':
            for m in [self.adapter_0_down, self.adapter_0_up]:
                for p in m.parameters():
                    p.requires_grad = True
            for m in [self.adapter_1_down, self.adapter_1_up]:
                for p in m.parameters():
                    p.requires_grad = False

        elif name == 'adapter_1':
            for m in [self.adapter_1_down, self.adapter_1_up]:
                for p in m.parameters():
                    p.requires_grad = True
            for m in [self.adapter_0_down, self.adapter_0_up]:
                for p in m.parameters():
                    p.requires_grad = False

        elif isinstance(name, list):
            for n in name:
                m = getattr(self, f'{n}_down')
                for p in m.parameters():
                    p.requires_grad = True
                m = getattr(self, f'{n}_up')
                for p in m.parameters():
                    p.requires_grad = True
        return

    def adapter_layer_forward_bert(self, hidden_states, input_tensor, layer_norm):
        hidden_states, residual = self.pre_forward(hidden_states, input_tensor, layer_norm)
        hidden_states = self.forward(hidden_states, residual)
        hidden_states = self.post_forward(hidden_states, input_tensor, layer_norm)
        return hidden_states

    def pre_forward(self, hidden_states, input_tensor, layer_norm):
        residual = hidden_states # residual_before_ln = True
        if layer_norm:
            hidden_states = layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states, residual

    def post_forward(self, hidden_states, input_tensor, layer_norm):
        if layer_norm:
            hidden_states = layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states

    def get_agg_out(self, outs, weights):
        agg_out = weights[:, :, 0].unsqueeze(-1) * outs[0]
        for i, out in enumerate(outs[1:]):
            agg_out += weights[:, :, i+1].unsqueeze(-1) * out
        return agg_out

    def forward(self, hidden_states, input_tensor):
        if not self.gating:
            # one adapter for all
            down = self.active_adapter_down(hidden_states)
            down = self.actv(down)
            up = self.active_adapter_up(down)

            hidden_states = input_tensor + up

        elif hasattr(self, 'adapter_2_down'):
            up_outs = []
            for i in [0, 2]:
                adapter_down = getattr(self, f'adapter_{i}_down')
                down_out = adapter_down(hidden_states)
                down_out = self.actv(down_out)
                adapter_up = getattr(self, f'adapter_{i}_up')
                up_out = adapter_up(down_out)
                up_outs.append(up_out)

            # weight_up = F.softmax(self.gating_module(hidden_states) + 10**-6, dim=-1)
            weight_up = torch.ones(list(up_out.shape)[:-1] + [2]).to('cuda') * 0.5
            agg_up_out = self.get_agg_out(up_outs, weight_up)
            hidden_states = input_tensor + agg_up_out * self.scaling

        else:
            # one gating for all
            up_outs = []
            for i in range(2):
                adapter_down = getattr(self, f'adapter_{i}_down')
                down_out = adapter_down(hidden_states)
                down_out = self.actv(down_out)
                adapter_up = getattr(self, f'adapter_{i}_up')
                up_out = adapter_up(down_out)
                up_outs.append(up_out)

            # weight_up = F.softmax(self.gating_module(hidden_states) + 10**-6, dim=-1)
            weight_up = torch.ones(list(up_out.shape)[:-1] + [2]).to('cuda') * 0.5
            agg_up_out = self.get_agg_out(up_outs, weight_up)
            hidden_states = input_tensor + agg_up_out * self.scaling
        return hidden_states