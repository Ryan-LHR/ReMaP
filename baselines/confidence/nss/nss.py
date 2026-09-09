"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Our Own Implementation
Reference Paper     : "Neuron Sensitivity-Guided Test Case Selection"

Note:
    This implementation is developed based on the descriptions and algorithms (e.g., pseudocode)
    provided in the original paper. No official code was released by the authors.

--------------------------------------------------------------------------------------
"""
import torch

from baselines.confidence.nss.nss_utils import *


def prioritize_by_nss(model, model_name, dataset_name, cand_loader, seed):
    """NSS"""

    'Step 0: Setup'
    k = 0.1  # ratio of selected sensitivity neurons, same as NSS
    # k = 0.5  # for test

    ratio = 0.1  # selected ratio of inputs for Sensitive Neuron Identifier, same as NSS
    # ratio = 0.0001
    mutation_types = [
        'shift',
        'rotation',
        'scale',
        'shear',
        'contrast',
        'brightness',
        'blur'
    ]  # same as NSS
    # mutation_types = ['gaussian_noise']  # for test

    params = {
        "shift": [(0.05, 0.15), (0.05, 0.15)],
        "rotation": (5, 15),
        "scale": ((0.8, 1.2), (0.8, 1.2)),
        "shear": [15, 30],
        "contrast": [0.5, 1.5],
        "brightness": [0.5, 1.5],
        "blur": [2, 7],
        # "gaussian_noise": {'mean': 0.0, 'std': 0.001}
    }

    # get batch_size
    first_batch = next(iter(cand_loader))
    batch_size = first_batch[0].size(0)

    'Step 1: Generate Inputs Pairs'
    # Random select inputs for identify
    selected_ori_set = create_subset_dataset(cand_loader, ratio, seed=seed)

    # Generate mutation inputs
    selected_mut_set = get_mutated_dataset(selected_ori_set, mutation_types, params)

    'Step 2: Identify Sensitive Neurons'
    # set last encoder layer as target layer
    if model_name == "MNIST-LeNet5":
        layer_name = 'pool2'
    elif model_name == "SVHN-VGG16":
        layer_name = 'pool5'
    elif model_name in ["FM-ResNet20", "C10-ResNet20"]:
        layer_name = 'layer3'
    elif model_name == "IM100Test-deit_base_patch16_224":
        layer_name = 'hf_model.classifier'
    else:
        raise ValueError("Model not found")
    print(f"Current target layer is: {layer_name}")
    sensitive_neurons = identity_sensitive_neurons(
        model, selected_ori_set, selected_mut_set, batch_size, layer_name, k, dataset_name
    )

    'Step 3: Calculate TNSScore'
    cand_set_ori = cand_loader.dataset
    # cand_set_ori = get_mutated_dataset(cand_set_ori, mutation_types, params)  # for test
    cand_set_mut = get_mutated_dataset(cand_set_ori, mutation_types, params)
    TNSScore = get_TNSScore(
        model, cand_set_ori, cand_set_mut, batch_size, layer_name, sensitive_neurons, dataset_name
    )

    'Step 4: Prioritize by TNSScore'
    _, prioritized_indices = torch.sort(TNSScore, descending=True)
    prioritized_indices = prioritized_indices.tolist()

    device = next(model.parameters()).device
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in cand_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            y_pred.extend(preds.numpy())
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_mask = (y_true == y_pred)
    wrong_mask = ~correct_mask

    print("Correct samples:", correct_mask.sum(),
          "Wrong samples:", wrong_mask.sum())

    print("Mean TNS(correct) =", float(TNSScore[correct_mask].mean()))
    print("Mean TNS(wrong)   =", float(TNSScore[wrong_mask].mean()))

    return prioritized_indices