import torch
import torch.nn as nn
import numpy as np
# from core.utils.network_util import initseq

class MLPs(nn.Module):
    def __init__(self, input_ch, output_ch, width, depth, actv_fn='relu', mlp_type='plane_jane', quantity_type='normal', 
                 geometric_init=True, weight_norm=True):
        super(MLPs, self).__init__()
        assert mlp_type in ['plane_jane', 'fully_fused']
        assert actv_fn in ['relu', 'leaky_relu', 'softplus']
        assert quantity_type in ['normal', 'opacity']
        
        

        self.output_ch = output_ch
        self.input_ch = input_ch
        self.actv_fn = actv_fn
        self.width = width
        self.depth = depth
        self.mlp_type = mlp_type
        self.quantity_type = quantity_type
        
        #! depth = # of hidden layers
        if mlp_type == 'plane_jane':
            # actv_fn = nn.ReLU() if actv_fn == 'relu' elif actv_fn == 'leaky_relu' else nn.Softplus() 
            actv_fn = nn.ReLU() if actv_fn == 'relu' else nn.LeakyReLU() if actv_fn == 'leaky_relu' else nn.Softplus(beta=100)
            layers = [nn.Linear(input_ch, width), actv_fn]
            if geometric_init:
                torch.nn.init.constant_(layers[-2].bias, 0.0)
                torch.nn.init.constant_(layers[-2].weight[:, 3:], 0.0)
            for i in range(depth):
                if i == depth - 1: #! last layer
                    layers += [nn.Linear(width, output_ch)]
                    if geometric_init:
                        torch.nn.init.normal_(layers[-1].weight, mean=np.sqrt(np.pi) / np.sqrt(output_ch), std=0.0001)
                        torch.nn.init.constant_(layers[-1].bias, 0.6)
                    if weight_norm:
                        layers[-1] = nn.utils.weight_norm(layers[-1])
                else:
                    layers += [nn.Linear(width, width), actv_fn]
                    if geometric_init:
                        torch.nn.init.constant_(layers[-2].bias, 0.0)
                        torch.nn.init.normal_(layers[-2].weight, 0.0, np.sqrt(2) / np.sqrt(width))
                    if weight_norm:
                        layers[-2] = nn.utils.weight_norm(layers[-2])
    

        elif mlp_type == 'fully_fused':
            try:
                import tinycudann as tcnn
            except ImportError:
                raise ImportError("tinycudann not found, plz check the installation of tinycudann")
            
            if actv_fn == 'relu':
                actv_fn = 'ReLU'
            elif actv_fn == 'leaky_relu':
                actv_fn = 'LeakyReLU'
            else:
                raise NotImplementedError
            
            prior_dict= {"network": {
                    "otype": "FullyFusedMLP",
                    "activation": actv_fn,
                    "output_activation": "None",
                    "n_neurons": width, #64,
                    "n_hidden_layers": depth
                    }}
            
            layers = [tcnn.Network(input_ch, width, prior_dict["network"])]
        else:
            raise NotImplementedError 
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output_quantity = self.layers(input)
        
        if self.quantity_type == 'opacity':
            output_quantity = torch.sigmoid(output_quantity)
            assert output_quantity.shape[-1] == 1
            
        elif self.quantity_type == 'normal':
            output_quantity = torch.tanh(output_quantity)
            output_quantity = output_quantity / output_quantity.norm(dim=-1, keepdim=True)
            assert output_quantity.shape[-1] == 3
            
        return output_quantity


       