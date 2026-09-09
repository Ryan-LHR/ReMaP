import torch
import torch.nn as nn
from copy import deepcopy

class NeuronInterventionLayer(nn.Module):
    """
    Define a replace layer for Linear Layer
    modify the value of given neuron
    """
    def __init__(self, original_layer, neuron_idx, mode, value=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.neuron_idx = neuron_idx
        self.mode = mode
        self.value = value

    def forward(self, x):
        out = self.original_layer(x)

        if self.mode == "inverse":  # inverse the neuron activation, e.g. 23 -> -23
            out[:, self.neuron_idx] = -out[:, self.neuron_idx]

        elif self.mode == "set_zero":  # set the neuron activation to zero, e.g. 23 -> 0
            out[:, self.neuron_idx] = 0.0
        else:
            raise ValueError
        return out



def get_mutate_model(model, m_type, seed, target_layer, num_neuron=1):
    """
    mutate model on target layer
    """
    modified_model = deepcopy(model)

     # (1) get new model with modified layer
    layer_orig = getattr(modified_model, target_layer)
    if not isinstance(layer_orig, nn.Linear):
        raise TypeError(f"Target layer {target_layer} is not nn.Linear, got {type(layer_orig)}")

    # (2) basic info
    device = layer_orig.weight.device
    out_dim = layer_orig.out_features
    in_dim = layer_orig.in_features

    # (3) Set random seed
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)

    # (4) Random select a neuron
    target_neuron = torch.randperm(out_dim, generator=g)[:num_neuron].tolist()
    print(f"target mutate neuron indices are: {target_neuron}")

    # (5) Mutation
    if m_type == 'GF':  # gauss noise to a neuron's weights
        gauss_mu = 0.0
        gauss_std = 0.1
        with torch.no_grad():
            for idx in target_neuron:
                noise = gauss_std * torch.randn(in_dim, generator=g) + gauss_mu
                noise = noise.to(device)
                layer_orig.weight.data[idx, :] += noise
        layer_new = layer_orig

    elif m_type == 'WS':  # shuffle the weights of a neuron
        with torch.no_grad():
            for idx in target_neuron:
                perm = torch.randperm(in_dim, generator=g)
                perm = perm.to(device)
                layer_orig.weight.data[idx, :] = layer_orig.weight.data[idx, perm]
        layer_new = layer_orig

    elif m_type == 'NAI':  # neuron activation inverse
        layer_new = NeuronInterventionLayer(layer_orig, target_neuron, mode="inverse")

    elif m_type == 'NEB':  # set neuron to zero
        layer_new = NeuronInterventionLayer(layer_orig, target_neuron, mode="set_zero")
    else:
        raise ValueError(f"model mutation type {m_type} not found!")

    setattr(modified_model, target_layer, layer_new)
    return modified_model