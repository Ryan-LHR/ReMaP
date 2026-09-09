# -*-coding:utf-8-*-
from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min
import torch

# art.config / art.utils are imported lazily inside _generate_rand_pool below: ART's declared pins break
# its import under newer scipy in some envs, and only certpri (an adversarial baseline) needs them, so we
# must not import ART at module load -- otherwise `from baselines import *` would crash every run.


def compute_class_gradient_pytorch(model, inputs, target_class, device):
    """
    Compute the gradient of the target class output with respect to the inputs.

    :param model: A trained PyTorch model.
    :param inputs: A batch of input samples as a NumPy array.
    :param target_class: The target class label for which to compute the gradient.


    :return: Gradients as a NumPy array.
    """
    model.eval()  # Set model to evaluation mode
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)

    # Move to the same device as the model
    if device is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    inputs_tensor = inputs_tensor.to(device).requires_grad_(True)

    # Forward pass
    outputs = model(inputs_tensor)

    # Select the output corresponding to the target class
    target_outputs = outputs[:, target_class]

    # Backward pass
    target_outputs.backward(torch.ones_like(target_outputs))

    # Extract gradients
    gradients = inputs_tensor.grad.detach().cpu().numpy()

    return gradients

def inverper_c(
    classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: float,
    c_init: float = 1.0,
    pool_factor: int = 10,
    y_pred=None,
    device=None,
) -> float:
    """
    Compute CLEVER score for a targeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: Initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Check if the targeted class is different from the predicted class
    # y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:  # pragma: no cover
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    rand_pool_grad_set = []
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples (ART imported here lazily -- see note at the top of the file)
    from art.config import ART_NUMPY_DTYPE
    from art.utils import random_sphere
    rand_pool = np.reshape(
        random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm),
        shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)

    # if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
    #     np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:  # pragma: no cover
        raise ValueError(f"Norm {norm} not supported")

    # Compute gradients for all samples in rand_pool
    for i in range(batch_size):
        rand_pool_batch = rand_pool[i * pool_factor : (i + 1) * pool_factor]

        # Compute gradients
        # grad_pred_class = classifier.class_gradient(rand_pool_batch, label=pred_class)
        grad_pred_class = compute_class_gradient_pytorch(classifier, rand_pool_batch, pred_class, device)

        if np.isnan(grad_pred_class).any() :  # pragma: no cover
            raise Exception("The classifier results NaN gradients.")

        grad = grad_pred_class
        grad = np.reshape(grad, (pool_factor, -1))
        grad = np.linalg.norm(grad, ord=norm, axis=1)
        rand_pool_grad_set.extend(grad)

    rand_pool_grads = np.array(rand_pool_grad_set)

    # Loop over the batches
    for _ in range(nb_batches):
        # Random selection of gradients
        grad_norm = rand_pool_grads[np.random.choice(pool_factor * batch_size, batch_size)]
        grad_norm = np.max(grad_norm)
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    # values = classifier.predict(np.array([x]))
    values = y_pred
    value = values[:, pred_class] - 0.5

    # Compute scores
#     score = np.min([-value[0] / loc, radius])
    score = -value[0] / loc

    return score