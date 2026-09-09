import random

import numpy as np
import torch
import torchvision.transforms.functional as F

def mutate_by_gaussian_noise(input, params):
    """
    mutate single tensor input by gaussian noise
    """
    mean = params['mean']
    std = params['std']

    mutated_input = input + torch.randn_like(input) * std + mean

    return mutated_input


def mutate_by_shift(x, param):
    """Shift image in x/y directions based on ratio ranges."""
    (sx_min, sx_max), (sy_min, sy_max) = param
    w, h = F.get_image_size(x)

    # sample shift ratio in given ranges
    sx = random.uniform(sx_min, sx_max) * random.choice([-1.0, 1.0])
    sy = random.uniform(sy_min, sy_max) * random.choice([-1.0, 1.0])

    tx = int(round(sx * w))
    ty = int(round(sy * h))

    # affine with translation only
    return F.affine(
        x,
        angle=0.0,
        translate=[tx, ty],
        scale=1.0,
        shear=[0.0, 0.0],
    )


def mutate_by_rotation(x, param):
    """Rotate image by a random angle within given range (degrees)."""
    deg_min, deg_max = param
    angle = random.uniform(deg_min, deg_max) * random.choice([-1.0, 1.0])
    return F.rotate(x, angle=angle)


def mutate_by_scale(x, param):
    """Scale image isotropically with scale factor within given ranges."""
    (sx_min, sx_max), (sy_min, sy_max) = param
    # here we use a single scalar scale to keep aspect ratio
    scale_x = random.uniform(sx_min, sx_max)
    scale_y = random.uniform(sy_min, sy_max)
    scale = (scale_x + scale_y) / 2.0

    return F.affine(
        x,
        angle=0.0,
        translate=[0, 0],
        scale=scale,
        shear=[0.0, 0.0],
    )


def mutate_by_shear(x, param):
    """Shear image horizontally with angle in given range."""
    s_min, s_max = param
    shear = random.uniform(s_min, s_max) * random.choice([-1.0, 1.0])
    return F.affine(
        x,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=[shear, 0.0],  # only horizontal shear
    )


def mutate_by_contrast(x, param):
    """Adjust contrast with factor in given range."""
    c_min, c_max = param
    factor = random.uniform(c_min, c_max)
    return F.adjust_contrast(x, factor)


def mutate_by_brightness(x, param):
    """Adjust brightness with factor in given range."""
    b_min, b_max = param
    factor = random.uniform(b_min, b_max)
    return F.adjust_brightness(x, factor)


def mutate_by_blur(x, param):
    """Blur image with Gaussian kernel size in given integer range."""
    k_min, k_max = param
    # torchvision GaussianBlur requires odd kernel size
    valid_ks = [k for k in range(k_min, k_max + 1) if k % 2 == 1]
    ks = random.choice(valid_ks)
    return F.gaussian_blur(x, kernel_size=[ks, ks])


class BenignMutation:
    """
    Benign mutation operator set used in NSS.
    Randomly selects ONE mutation type and applies it using torchvision ops.
    """

    def __init__(self, mutation_types, params):
        print(f"Current mutation types contain: {mutation_types}")
        self.mutation_types = mutation_types
        self.params = params

    def __call__(self, x):
        x = self._to_torch_image(x)
        type = random.choice(self.mutation_types)

        if type == 'shift':
            x = mutate_by_shift(x, self.params['shift'])
        elif type == 'rotation':
            x = mutate_by_rotation(x, self.params['rotation'])
        elif type == 'scale':
            x = mutate_by_scale(x, self.params['scale'])
        elif type == 'shear':
            x = mutate_by_shear(x, self.params['shear'])
        elif type == 'contrast':
            x = mutate_by_contrast(x, self.params['contrast'])
        elif type == 'brightness':
            x = mutate_by_brightness(x, self.params['brightness'])
        elif type == 'blur':
            x = mutate_by_blur(x, self.params['blur'])
        elif type == 'gaussian_noise':
            x = mutate_by_gaussian_noise(x, self.params['gaussian_noise'])
        else:
            raise ValueError(f"Unknown mutation type: {type}")

        return x

    def _to_torch_image(self, x):
        # numpy -> torch
        if isinstance(x, np.ndarray):  # convert for ImageNet
            t = torch.from_numpy(np.ascontiguousarray(x))
            if t.dtype == torch.uint8:
                t = t.float() / 255.0
            else:
                t = t.float()
            return t
        return x