import numpy as np
# from PIL import Image
# import time
import os
# import sys
# import datetime
# import keras
# import math
# from keras.models import Model
# import random
# from keras.datasets import mnist
# from numpy import arange
# import argparse
# from keras.applications import vgg19,resnet50

# from keras.applications.vgg19 import preprocess_input
# import tensorflow as tf

# from datautils import get_data,get_model,data_proprecessing


def generate_ratio_vector(num, ratio):
    import math
    perturbate_num = math.ceil(num * ratio)
    non_perturbate_num = num - perturbate_num
    a = np.zeros(perturbate_num) + 1
    b = np.zeros(non_perturbate_num)
    a_b = np.concatenate((a, b), axis=0)
    np.random.shuffle(a_b)
    return a_b


def black(image, i=0, j=0, interval=2):
    image = np.array(image, dtype=float)
    image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j] = 0
    return image.copy()


def white(image, i=0, j=0, interval=2):
    image = np.array(image, dtype=float)
    image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j] = 255
    return image.copy()


def reverse_color(image, i=0, j=0, interval=2):
    image = np.array(image, dtype=float)
    part = image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j].copy()
    reversed_part = 255 - part
    image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j] = reversed_part
    return image

def gauss_noise(image,i=0,j=0,mean=0, var=0.1,ratio=1.0, interval=2):
    image = np.array(image, dtype=float)
    image = image.astype('float32') / 255
    part = image[0+interval*i:interval+interval*i,0+interval*j:interval+interval*j].copy()
    ratio_vector = generate_ratio_vector(len(part.ravel()),ratio).reshape(part.shape)
    noise = np.random.normal(mean, var ** 0.5, part.shape)
    noise = noise * ratio_vector
    image[0+interval*i:interval+interval*i,0+interval*j:interval+interval*j] += noise
    image = np.clip(image, 0, 1)
    image *= 255
    return image.copy()

def shuffle_pixel(image, i=0, j=0, interval=2):
    image = np.array(image, dtype=float)
    # image /= 255
    part = image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j].copy()
    part_r = part.reshape(-1, 1)
    np.random.shuffle(part_r)
    part_r = part_r.reshape(part.shape)
    image[0 + interval * i:interval + interval * i, 0 + interval * j:interval + interval * j] = part_r
    return image
    

# exp_id = sys.argv[1]
# perturbate_type = sys.argv[2]
from tqdm import tqdm
import gc

def perturb_image(basedir, perturbate_type, dataloader, model_name):
    print(f"Input Mutation")
    # x,y = get_data(exp_id)
    x, _ = collect_x_y_from_dataloader(dataloader)

    # making needed directory
    mutation_dir = os.path.join(basedir, "mutated_input", str(perturbate_type))
    if not os.path.exists(mutation_dir):
        os.makedirs(mutation_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(mutation_dir) if f.endswith(".npy")]
    if len(existing_files) >= len(x):
        print(f"[SKIP] {perturbate_type}: found {len(existing_files)} npy files, len(x)={len(x)}, skip mutation.")
        return  # 或者直接 `continue`，看你是在函数里还是在外层 for 里调用的


    image = x[0]
    # H, W = image.shape[:2]
    H, W, C = image.shape

    # Change patch and interval for different image size
    grid_h, grid_w = 16, 16  # 固定生成 256 个 mutants
    # patch_h = H // grid_h  # 每个 patch 的高
    # patch_w = W // grid_w  # 每个 patch 的宽
    interval = H // grid_h  # 对32x32是2，对28x28是1，对224x224是14
    if model_name in ["SVHN-VGG16"]:
        grid_h, grid_w = 12, 12  # 固定生成 144 个 mutants(less mutants for limited disk volume)

    image_id = 0 
    # for image in x:
    for image in tqdm(x, desc=f"Perturbating ({perturbate_type})"):
        tt_temp = []
        # for i in range(16):
        #     for j in range(16):
        for i in range(grid_h):
            for j in range(grid_w):
                if perturbate_type == 'gauss':
                    tt = gauss_noise(image, i, j, ratio=1.0, var=0.01, interval=interval)
                elif perturbate_type == 'white':
                    tt = white(image, i, j, interval=interval)
                elif perturbate_type == 'black':
                    tt = black(image, i, j, interval=interval)
                elif perturbate_type == 'reverse':
                    tt = reverse_color(image, i, j, interval=interval)
                elif perturbate_type == 'shuffle':
                    tt = shuffle_pixel(image, i, j, interval=interval)
                # tt_temp.append(data_proprecessing(exp_id)(tt))
                tt_temp.append(return_test_images(tt))

        # arr = np.array(tt_temp).reshape(-1, 32, 32, 3)
        arr = np.array(tt_temp).reshape(-1, H, W, C)
        np.save(os.path.join(mutation_dir, str(image_id) + '.npy'), arr)
        image_id += 1

        del tt_temp
        del arr
        del image


        if image_id % 1000 == 0:
            # print(str(image_id))
            gc.collect()
            # print(time.time()-start)
    # print(time.time()-start)
    print('finish generating...')

def return_test_images(test_images):
    test_images = test_images.astype('float32') / 255
    return test_images


def collect_x_y_from_dataloader(dataloader):
    all_x = []
    all_y = []

    for inputs, labels in dataloader:
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()

        # (B, C, H, W) → (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1)

        # Check
        # [0,1]
        if inputs.max() <= 1.0 and inputs.min() >= 0.0:
            inputs_np = (inputs.numpy() * 255.0).astype(np.float32)

        # [0, 255]
        elif inputs.max() > 1.0 and inputs.max() <= 255:
            inputs_np = inputs.numpy().astype(np.float32)

        else:
            raise ValueError(
                f"Detected standardized input: range=({inputs.min()}, {inputs.max()}).\n"
                "Your dataset uses transforms.Normalize, so you MUST unnormalize before using PRIMA."
            )


        all_x.append(inputs_np)
        all_y.append(labels.numpy())

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    return x, y
