from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

# Define model modules included neuron units
INCLUDED_MODULES = (
    nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.MaxPool2d,
    nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    nn.Linear,
)

EXCLUDED_NAMES = [
    'transformer_encoder',  # need more than 120GB RAM, so excluded
    'token_mixer',
    '.proj.',
    'patch_embed',
    'network.',
    'query',  # need more than 120GB RAM, so excluded
    'key',
    'value',
    'hf_model.vit',  # need more than 80GB RAM, so excluded
    # 'hf_model.vit.encoder.layer.0.',
    # 'hf_model.vit.encoder.layer.1.',
    # 'hf_model.vit.encoder.layer.2.',
    # 'hf_model.vit.encoder.layer.3.',
    # 'hf_model.vit.encoder.layer.4.',
    # 'hf_model.vit.encoder.layer.5.',
    # 'hf_model.vit.encoder.layer.6.',
    # 'hf_model.vit.encoder.layer.7.',
    # 'hf_model.vit.encoder.layer.8.',
    # 'hf_model.vit.encoder.layer.9.',
    # 'hf_model.vit.encoder.layer.10.',
    'audio_spectrogram_transformer.encoder.layer.0.',  # need more than 80GB RAM, so excluded front layers
    'audio_spectrogram_transformer.encoder.layer.1.',
    'audio_spectrogram_transformer.encoder.layer.2.',
    'audio_spectrogram_transformer.encoder.layer.3.',
    'audio_spectrogram_transformer.encoder.layer.4.',
    'audio_spectrogram_transformer.encoder.layer.5.',
    'audio_spectrogram_transformer.encoder.layer.6.',
    'audio_spectrogram_transformer.encoder.layer.7.',
    'audio_spectrogram_transformer.encoder.layer.8.',
    'audio_spectrogram_transformer.encoder.layer.9.',
    'audio_spectrogram_transformer.encoder.layer.10.',
]

def get_method_and_strategy(s: str):
    """
    get method and strategy from method name
    """
    if '-' not in s:
        return None

    method, strategy = s.rsplit('-', 1)

    # strategy is 'cam' or 'ctm'
    if strategy not in ("ctm", "cam"):
        return None

    print(f'\ncurrent coverage is {method}, strategy is {strategy}')
    return method, strategy

def forward_hook_count(neurons_dict, layer_name, module, input, output):
    """
    Forward hook to record the number of neurons in each layer.
    Counts neurons by channels for Conv and Pool layers, and by output features for Linear layers.
    """
    if isinstance(module, nn.Linear):
        num_neurons = output.size(-1)  # Linear: features are the last dim (2D or 3D transformer output)
    elif isinstance(module, INCLUDED_MODULES):
        num_neurons = output.size(1)  # Conv/Pool: count by channels
    else:
        return  # Ignore other layer types
    neurons_dict[layer_name] = num_neurons

def count_neurons(model, dataloader, device=None):
    """
    Count the number of neurons (count by channels)
    get input size through dataloder
    """
    first_batch = next(iter(dataloader))
    inputs = first_batch[0].to(device)
    if inputs.dim() == 4:
        # convert [batch_size, channels, height, width] to [1, channels, height, width]
        input_tensor = inputs[0].unsqueeze(0)

    elif inputs.dim() == 3:
        # point cloud [B, N, 3] -> [1, N, 3]; DGCNN.forward accepts (B, N, 3) and transposes internally
        input_tensor = inputs[0].unsqueeze(0)

    elif inputs.dim() == 2 and inputs.is_floating_point():
        # audio waveform [B, T]: feed as-is and keep batch>=2
        if inputs.size(0) >= 2:
            input_tensor = inputs
        else:
            input_tensor = inputs.expand(2, inputs.size(1)).contiguous()

    else:
        raise ValueError(f"unsupported input dimensions: {inputs.dim()}")
    input_size = tuple(input_tensor.size()[1:])  # (channels, height, width) or (channels, length)

    neurons_dict = {}
    hooks = []
    total_modules = list(model.named_modules())
    focused_models = []
    for name, module in model.named_modules():     # Register hooks to relevant layers
        if module != model:
            if isinstance(module, INCLUDED_MODULES):
                if any(e in name for e in EXCLUDED_NAMES):
                    continue
                hook = module.register_forward_hook(partial(forward_hook_count, neurons_dict, name))
                hooks.append(hook)
                focused_models.append(module)

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:  # remove hook
        h.remove()

    total_neurons = sum(neurons_dict.values())
    print(f"\nDetected input size: {input_size}"
          f"\nNeurons per layer: {neurons_dict}"
          f"\nTotal neurons: {total_neurons}\n")
    return total_neurons

def forward_hook_activations(act_values, layer_name, module, input, output):
    """
    Forward hook to record activation values of neurons in each layer.
    For Linear layers, store values per output feature.
    For Conv/Pool layers, compute mean over spatial dimensions to store a single value per channel.
    """
    # get values
    if isinstance(module, nn.Linear):
        # Linear output is [batch, features], or [batch, seq/time, features] for transformer Linear
        # (e.g. AST encoder). Mean-pool the seq/time dim so we keep one value per feature per sample;
        # otherwise storing every (sample, seq, feature) triple blows up memory (OOM). Pool on-device
        # before the .cpu() transfer to also cap the per-batch peak.
        out = output.detach()
        if out.dim() == 3:
            out = out.mean(dim=1)
        vals = out.cpu().numpy()

    else:
        # For Conv/Pool layers: output: [batch, channels, height, width]
        # Compute mean over spatial dims to keep a single value per channel
        if output.dim() == 4:
            vals = output.mean(dim=[2, 3]).detach().cpu().numpy()

        elif output.dim() == 3:
            # Conv1d/Pool1d: [N,C,L] -> mean over L；Embedding/LSTM: [N,T,D] -> mean over T
            if isinstance(module, (nn.Conv1d, nn.BatchNorm1d,
                                   nn.AvgPool1d, nn.MaxPool1d,
                                   nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
                vals = output.mean(dim=2).detach().cpu().numpy()

        elif output.dim() == 2:
            # [batch, features]
            vals = output.detach().cpu().numpy()

        else:
            raise ValueError(f"unsupported output dimensions: {output.dim()}")
            # If for some reason it's not 4D (1D conv?), directly use channels dimension
            # vals = output.detach().cpu().numpy()

    # Add values to dict
    if layer_name not in act_values:
        act_values[layer_name] = {}
    # Iterate through each neuron
    for i in range(vals.shape[1]):
        neuron_id = i + 1  # neuron id start from 1 (not 0)
        if neuron_id not in act_values[layer_name]:
            act_values[layer_name][neuron_id] = []
        # Append all batch samples' values for this neuron
        act_values[layer_name][neuron_id].extend(vals[:, i].tolist())

def get_activation_values(model, dataloader, device=None):
    """get activation values of all neurons"""
    act_values = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, INCLUDED_MODULES) and module != model:
            if any(e in name for e in EXCLUDED_NAMES):
                continue
            hook = module.register_forward_hook(partial(forward_hook_activations, act_values, name))
            hooks.append(hook)

    # Perform forward passes to trigger hooks
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Get activation values"):
            inputs = batch[0].to(device)
            _ = model(inputs)

    # Perform first batch for test
    # first_batch = next(iter(dataloader))
    # inputs = first_batch[0]
    # _ = model(inputs)

    for h in hooks:
        h.remove()
    print(f"Collected neuron activation values\n")
    return act_values

def get_sample_activations(act_values, s):
    """
    Given act_values and a sample index s, return the activation values of that sample.
    act_values format:
        {layer_name: {neuron_id: [val_sample_0, val_sample_1, ...]}}
    output format:
        {layer_name: {neuron_id: val_sample_s}}
    """
    sample_act = {}
    for layer_name in act_values:
        sample_act[layer_name] = {}
        for neuron_id in act_values[layer_name]:
            sample_act[layer_name][neuron_id] = act_values[layer_name][neuron_id][s]
    return sample_act

def get_activation_ranges(model, train_loader, device=None):
    """get activation ranges of all neurons through train_loader"""
    # firstly, get activation values of train_loader
    act_values = get_activation_values(model, train_loader, device)

    act_ranges = {}
    for layer_name, neurons in act_values.items():
        act_ranges[layer_name] = {}
        for neuron_id, samples in neurons.items():
            act_ranges[layer_name][neuron_id] = (min(samples), max(samples))

    print(f"Collected neuron activation ranges")
    return act_ranges

def get_k_ranges(act_ranges, k):
    """get k ranges(multisections) of all neurons through act_ranges"""
    act_k_ranges = {}
    for layer_name, neurons in act_ranges.items():
        act_k_ranges[layer_name] = {}
        for neuron_id, (min_val, max_val) in neurons.items():
            interval_width = (max_val - min_val) / k
            # Generate k intervals
            intervals = [min_val + i * interval_width for i in range(k + 1)]  # k+1
            act_k_ranges[layer_name][neuron_id] = intervals

    print(f"Collected k activation ranges")
    return act_k_ranges