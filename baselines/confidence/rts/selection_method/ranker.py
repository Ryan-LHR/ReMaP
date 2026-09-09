from typing import List

# import faiss
# import fast_pytorch_kmeans
import imagehash
# import mkl
import torch
torch.backends.cudnn.deterministic = True
import numpy as np

# import keras
# from keras import Model
# from keras import backend as K
# from keras_preprocessing.image import array_to_img
from PIL import Image
# from tqdm import tqdm

# import BestSolution1
# import BestSolution_filter
# import BestSolution_var
# import OOD_detection
# import general_util
# import tf_util
# from ATS.ATS import ATS
from baselines.confidence.rts.BestSolution import obtain_ssim, conquer, ob_sus_correct, get_noise_threshold
# from selection_method.necov_method import metrics
# from selection_method.rank_method.CES.condition import CES_ranker
# from utils import model_conf

# mkl.get_max_threads()
def array_to_img_custom(x, data_format='channels_last', scale=True):
    """
    将 NumPy 数组转换为 PIL Image 对象，类似于 Keras 的 array_to_img。

    参数:
        x (np.ndarray): 输入的 NumPy 数组，形状为 (H, W, C) 或 (C, H, W)。
        data_format (str): 图像的通道顺序，'channels_last' 或 'channels_first'。默认 'channels_last'。
        scale (bool): 是否缩放图像像素值到 [0, 255]。默认 True。

    返回:
        PIL.Image.Image: 转换后的 PIL Image 对象。
    """
    if data_format == 'channels_first':
        # 将通道维度转换为最后一维
        x = np.transpose(x, (1, 2, 0))

    if scale:
        # 缩放图像像素值到 [0, 255]
        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)  # 归一化到 [0, 1]
        else:
            x = np.zeros_like(x)  # 如果 x_max == x_min，则全设为0
        x = (x * 255).astype('uint8')
    else:
        # 确保图像像素值为 uint8
        if x.dtype != 'uint8':
            x = x.astype('uint8')

    # 创建 PIL Image 对象
    if x.shape[2] == 1:
        # 灰度图像
        x = x.squeeze(axis=2)  # 去掉单通道维度
        image = Image.fromarray(x, mode='L')
    elif x.shape[2] == 3:
        # RGB 图像
        image = Image.fromarray(x, mode='RGB')
    elif x.shape[2] == 4:
        # RGBA 图像
        image = Image.fromarray(x, mode='RGBA')
    else:
        raise ValueError(f"Unsupported number of channels: {x.shape[2]}")

    return image

class Ranker(object):
    def __init__(self, model, x):
        self.model = model
        self.x = x

    # def dac_rank_va1(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):
    #
    #     select_pro = self.model.predict(self.x)
    #     select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
    #     train_pro = self.model.predict(x_train)
    #
    #     x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
    #     x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])
    #
    #     ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
    #                              y_train=y_train,
    #                              x_select_hash=x_select_hash,
    #                              x_train_hash=x_train_hash)
    #
    #     suspicious_array, correct_array = BestSolution_var.ob_sus_correct_var1(sim_tolerate_num=sim_number,
    #                                                                            ssim_datas=ssim_datas,
    #                                                                            select_pro=select_pro,
    #                                                                            select_lable=select_lable,
    #                                                                            train_pro=train_pro,
    #                                                                            y_train=y_train)
    #
    #     temp = [suspicious_array, correct_array]
    #
    #     rank_list = []
    #
    #     for indexs in temp:
    #         # 获取的是索引
    #         scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
    #         # 获取索引+不确定度
    #         rank_list += scores
    #
    #     rank_list = np.array(rank_list).astype(np.int)
    #
    #     return rank_list
    #
    # def dac_rank_va2(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):
    #
    #     select_pro = self.model.predict(self.x)
    #     select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
    #
    #     x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
    #     x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])
    #
    #     noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
    #                                           y_train=y_train, x_train_hash=x_train_hash)
    #
    #     ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
    #                              y_train=y_train,
    #                              x_select_hash=x_select_hash,
    #                              x_train_hash=x_train_hash)
    #
    #     suspicious_array, noise_array = BestSolution_var.ob_sus_correct_var2(ssim_datas=ssim_datas,
    #                                                                          select_pro=select_pro,
    #                                                                          noise_threshold=noise_threshold)
    #
    #     temp = [suspicious_array, noise_array]
    #
    #     rank_list = []
    #
    #     for indexs in temp:
    #         # 获取的是索引
    #         scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
    #         # 获取索引+不确定度
    #         rank_list += scores
    #
    #     rank_list = np.array(rank_list).astype(np.int)
    #
    #     return rank_list
    #
    # def dac_rank_va3(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):
    #
    #     select_pro = self.model.predict(self.x)
    #     select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
    #     train_pro = self.model.predict(x_train)
    #
    #     x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
    #     x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])
    #
    #     noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
    #                                           y_train=y_train, x_train_hash=x_train_hash)
    #
    #     ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
    #                              y_train=y_train,
    #                              x_select_hash=x_select_hash,
    #                              x_train_hash=x_train_hash)
    #
    #     suspicious_array, noise_array, correct_array = ob_sus_correct(sim_tolerate_num=sim_number,
    #                                                                   ssim_datas=ssim_datas,
    #                                                                   select_pro=select_pro, select_lable=select_lable,
    #                                                                   train_pro=train_pro, y_train=y_train,
    #                                                                   noise_threshold=noise_threshold,
    #                                                                   filter_noise=False)
    #
    #     temp = [suspicious_array, noise_array, correct_array]
    #
    #     rank_list = []
    #
    #     for indexs in temp:
    #         # 获取的是索引
    #         scores = BestSolution_var.conquer(indexs=indexs)
    #         # 获取索引+不确定度
    #         rank_list += scores
    #
    #     rank_list = np.array(rank_list).astype(np.int)
    #
    #     return rank_list

    def dac_rank(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5,
                 train_pro=None, select_pro=None, seed=None):

        # select_pro = self.model.predict(self.x)  # original code
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        # train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img_custom(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img_custom(i)).hash.flatten() for i in x_train])

        # x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        # x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
                                              y_train=y_train, x_train_hash=x_train_hash, device=device, seed=seed)

        ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
                                 y_train=y_train,
                                 x_select_hash=x_select_hash,
                                 x_train_hash=x_train_hash, device=device, seed=seed)

        suspicious_array, noise_array, correct_array = ob_sus_correct(sim_tolerate_num=sim_number,
                                                                      ssim_datas=ssim_datas,
                                                                      select_pro=select_pro, select_lable=select_lable,
                                                                      train_pro=train_pro, y_train=y_train,
                                                                      noise_threshold=noise_threshold,
                                                                      filter_noise=False, device=device, seed=seed)

        temp = [suspicious_array, noise_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
            # 获取索引+不确定度
            rank_list += scores

        # rank_list = np.array(rank_list).astype(np.int)
        rank_list = np.array(rank_list).astype(np.int32)

        return rank_list


