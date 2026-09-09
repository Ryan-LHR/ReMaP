"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation  
Reference Paper     : "FAST: Boosting Uncertainty-based Test Prioritization Methods 
                       for Neural Networks via Feature Selection"

Note:
    This code is adapted from the official implementation provided by the authors.
    Minor modifications may have been made for compatibility or integration purposes.
--------------------------------------------------------------------------------------
"""

from baselines.confidence.simple_confidence import prioritize_by_deepgini
from baselines.confidence.fast.fast_utils import *
from utils.dufp.dufp_utils import get_labels_and_classes


def prioritize_by_fast(model, model_name, train_loader, train_set, train_vectors,
                       train_correct, cand_loader, cand_labels, batch_size, args):
    """
    FAST (official implementation of FAST)
    """

    'Step 1: Setup'
    # (1) Hyper parameters
    r = 0.05  # same as FAST (i.e., 5%)
    # r = 0  # namely the original probability
    conf_thres = 0.9  # same as FAST
    uncertainty_type = 'deepgini'  # same as NNS & FAST
    # batch_size = 1024
    # batch_size = 256

    # get true labels
    y_train, y_cand = get_labels_and_classes(
        args,
        model_name,
        train_loader,
        cand_loader,
    )

    # get num of classes
    classes = np.unique(y_train)
    num_classes = classes.size

    # Set Layer for denoising
    if model_name == "MNIST-LeNet5":
        layer_name = 'fc1'
        # layer_name = 'fc2'
    elif model_name == "SVHN-VGG16":
        layer_name = 'fc1'
        # layer_name = 'fc2'
    elif model_name in ["FM-ResNet20", "C10-ResNet20"]:
        layer_name = 'fc1'
    elif model_name == "IM100Test-deit_base_patch16_224":
        layer_name = 'hf_model.classifier'
    elif model_name == "ModelNet40-DGCNN":
        layer_name = 'dp2'  # nn.Dropout(0.5) -- identity in eval; remaining = linear3 + softmax
    elif model_name == "ESC50-AST":
        layer_name = 'model.classifier.layernorm'  # ASTMLPHead: layernorm -> dense; remaining = dense + softmax
    else:
        raise ValueError("Model not found")

    # (2) Construct trusted dataset
    loaders_dict, class_to_indices = construct_trusted_subset(
        train_set, y_train, train_vectors, train_correct,
        conf_thres, batch_size, args
    )
    # (3) Get output probabilities
    train_prob = np.array([tensor.cpu().numpy() for tensor in train_vectors])
    prob_dict = construct_prob_dict(class_to_indices, train_prob)

    # (4) Others
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    'Step 2: Call the official implementation of FAST'
    # (1) Initialize FAST
    fast = FAST(model_name, model=model, layer_index=None, layer_name=layer_name,
                classes=num_classes, device=device)

    # (2) Measure contribution
    fast.fetch_ns_torch(loaders_dict, prob_dict)

    # (3) Feature selection
    fast.fetch_class_patterns_ns(p=1-r)

    # (4) Get dataloader of purified probabilities
    prob_new_loader = fast.get_purified_dataloader(cand_loader, cand_labels)

    'Step 3: Prioritize by Uncertainty'
    if uncertainty_type == 'deepgini':
        prioritized_indices = prioritize_by_deepgini(prob_new_loader, device)
    else:
        raise ValueError("Uncertainty Type not found")

    return prioritized_indices
