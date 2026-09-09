import numpy as np
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input
import os
# import keras
import sys
# from datautils import get_data,get_model,data_proprecessing
from pathlib import Path

from tqdm import tqdm


def cos_distribution(cos_array):
    cos_distribute = [0 for i in range(10)]
    for i in cos_array:
        if i >= 0 and i < 0.1:
            cos_distribute[0] += 1
        elif i >= 0.1 and i < 0.2:
            cos_distribute[1] += 1
        elif i >= 0.2 and i < 0.3:
            cos_distribute[2] += 1
        elif i >= 0.3 and i < 0.4:
            cos_distribute[3] += 1
        elif i >= 0.4 and i < 0.5:
            cos_distribute[4] += 1
        elif i >= 0.5 and i < 0.6:
            cos_distribute[5] += 1
        elif i >= 0.6 and i < 0.7:
            cos_distribute[6] += 1
        elif i >= 0.7 and i < 0.8:
            cos_distribute[7] += 1
        elif i >= 0.8 and i < 0.9:
            cos_distribute[8] += 1
        elif i >= 0.9 and i <= 1.0:
            cos_distribute[9] += 1
    return cos_distribute



def model_feature_extraction(model_name, save_dir, m_type):
    """
    针对“模型变异”（GF/NEB/NAI/WS）部分，基于已经保存的：
      - 原始模型预测概率 ori_prob: predict_probability_vector_{model_name}.npy
      - 各变异模型预测结果: {model_name}_temp_result_{m_type}/{i}.npy
    计算 PRIMA 中的模型变异特征，并写入 {model_name}_{m_type}_feature.txt
    """

    ptype = m_type

    # 1) 规范化路径
    basedir = Path(save_dir)
    basedir.mkdir(parents=True, exist_ok=True)

    # 2) 原始模型的预测概率文件（已经由别的步骤算好并保存）
    predicting_file_path = basedir / "predict_prob" / f"predict_probability_vector.npy"
    if not predicting_file_path.exists():
        raise FileNotFoundError(
            f"[ERROR] ori_prob not found: {predicting_file_path}\n"
            f"请确保已先对原始模型计算并保存预测概率。"
        )

    ori_prob = np.load(predicting_file_path)  # shape: (N, num_classes)
    samples = ori_prob.shape[0]

    # 3) 特征文件路径，如 ResNet20_GF_feature.txt
    feature_txt_path = basedir / "features" /f"{model_name}_{ptype}_feature.txt"
    # 如果已经存在，直接跳过，避免重复写入
    if feature_txt_path.exists():
        print(f"[SKIP] Model features already extracted: {feature_txt_path}")
        return

    # 4) 初始化各类统计量
    euler = [0.0 for _ in range(samples)]
    mahat = [0.0 for _ in range(samples)]
    qube = [0.0 for _ in range(samples)]
    cos = [0.0 for _ in range(samples)]
    difference = [0.0 for _ in range(samples)]
    cos_list = [[] for _ in range(samples)]
    different_class = [[] for _ in range(samples)]

    # 5) 读取各个变异模型的预测结果：
    #    目录结构：basedir / f"{model_name}_temp_result_{ptype}" / "0.npy", "1.npy", "2.npy"
    temp_dir = basedir/ "predict_prob" / f"model_temp_result_{ptype}"

    for ptype in [ptype]:
        file_path = temp_dir / f"{ptype}.npy"
        if not file_path.exists():
            raise FileNotFoundError(
                f"[ERROR] mutated model prediction not found: {file_path}\n"
                f"请确认 core_prioritization_unit 是否已对该变异模型跑完预测。"
            )

        perturbated_prediction = np.load(file_path)  # shape: (samples, num_classes)
        if perturbated_prediction.shape[0] != samples:
            raise ValueError(
                f"[ERROR] shape mismatch: ori_prob has {samples} samples, "
                f"but {file_path} has {perturbated_prediction.shape[0]} samples."
            )

        # 遍历每个样本，累计 3 个变异模型带来的差异
        for ii in range(samples):
            pro = perturbated_prediction[ii]
            opro = ori_prob[ii]
            max_value_pos = np.argmax(opro)
            max_value = np.max(opro)

            # difference: 关注原最大概率类别上，概率变化的绝对差
            difference[ii] += abs(max_value - pro[max_value_pos])

            # 各种范数差
            euler[ii] += np.linalg.norm(pro - opro)
            mahat[ii] += np.linalg.norm(pro - opro, ord=1)
            qube[ii] += np.linalg.norm(pro - opro, ord=np.inf)

            # cosine distance
            denom = (np.linalg.norm(pro) * np.linalg.norm(opro))
            if denom == 0:
                co = 0.0
            else:
                co = 1 - (np.dot(pro, opro.T) / denom)
            co = min(max(co, 0.0), 1.0)  # clamp 到 [0,1]

            cos[ii] += co
            cos_list[ii].append(co)

            # 记录“预测类别发生改变”的情况
            if np.argmax(pro) != max_value_pos:
                different_class[ii].append(np.argmax(pro))

    # 6) 将特征写入 txt
    with open(feature_txt_path, 'a+', encoding='utf-8') as result_recording_file:
        for i in tqdm(range(samples), desc="model feature extraction"):
            dic = {}
            for key in different_class[i]:
                dic[key] = dic.get(key, 0) + 1
            wrong_class_num = len(dic)
            max_class_num = max(dic.values()) if dic else 0
            cos_dis = cos_distribution(cos_list[i])

            # print('id:', i)

            result_recording_file.write('image_id:' + str(i) + '\n')
            result_recording_file.write('euler:' + str(euler[i]) + '\n')
            result_recording_file.write('mahat:' + str(mahat[i]) + '\n')
            result_recording_file.write('qube:' + str(qube[i]) + '\n')
            result_recording_file.write('cos:' + str(cos[i]) + '\n')
            result_recording_file.write('difference:' + str(difference[i]) + '\n')
            result_recording_file.write('wnum:' + str(wrong_class_num) + '\n')
            result_recording_file.write('num_mc:' + str(max_class_num) + '\n')
            result_recording_file.write('fenbu:' + str(cos_dis) + '\n')

    print(f"[DONE] Model feature extraction saved to:\n  {feature_txt_path}")

# if __name__ == '__main__':
# def model_feature_extraction(model_name, save_dir, m_type):
#
#     # exp_id = sys.argv[1]
#     ptype = m_type
#     # sample = len(get_data(exp_id)[0])
#
#     # basedir = os.path.dirname(__file__)
#     # basedir = os.path.join(basedir, 'model')
#     # basedir = os.path.join(basedir, model_name)
#     basedir = save_dir
#     predicting_file_path = os.path.join(basedir, 'predict_probability_vector_'+str(model_name)+'.npy')
#     X_test,Y_test = get_data(model_name)
#     origin_model = get_model(model_name)
#     X_test = data_proprecessing(model_name)(X_test)
#     if not os.path.exists(predicting_file_path):
#         a = origin_model.predict(X_test)
#         # a = np.argmax(a, axis=1)
#         np.save(predicting_file_path,a)
#         ori_prob = a
#     else:
#         ori_prob = np.load(predicting_file_path)
#
#     file_name = 'image_perturbation_'+exp_id+'_'+ptype
#     result = np.argmax(ori_prob, axis=1)
#
#     samples = int(float(sample))
#     file_name = model_name+'_'+ptype+'_feature'
#     file_name = os.path.join(basedir, file_name)
#     euler = [0 for i in range(samples)]
#     mahat = [0 for i in range(samples)]
#     qube = [0 for i in range(samples)]
#     cos = [0 for i in range(samples)]
#     difference = [0 for i in range(samples)]
#     cos_list = [[] for i in range(samples)]
#     different_class = [[] for i in range(samples)]
#
#     for i in range(0, 3):
#
#         file_path = exp_id+'_temp_result_'+ptype+'/' + str(i) + '.npy'
#         file_path = os.path.join(basedir, file_path)
#
#         perturbated_prediction = np.load(file_path)
#         for ii in range(samples):
#             pro = perturbated_prediction[ii]
#             opro = ori_prob[ii]
#             max_value_pos = np.argmax(opro)
#             max_value = np.max(opro)
#             difference[ii] += abs(max_value - pro[max_value_pos])
#             euler[ii] += np.linalg.norm(pro - opro)
#             mahat[ii] += np.linalg.norm(pro - opro, ord=1)
#             qube[ii] += np.linalg.norm(pro - opro, ord=np.inf)
#             co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
#             if co < 0:
#                 co = 0
#             elif co > 1:
#                 co = 1
#             cos[ii] += co
#             cos_list[ii].append(co)
#
#             if np.argmax(pro) != max_value_pos:
#                 different_class[ii].append(np.argmax(pro))
#
#     result_recording_file = open(file_name + '.txt', 'a+')
#     for i in range(samples):
#
#         dic = {}
#         for key in different_class[i]:
#             dic[key] = dic.get(key, 0) + 1
#         wrong_class_num = len(dic)
#         if len(dic)>0:
#             max_class_num = max(dic.values())
#         else :
#             max_class_num = 0
#         cos_dis = cos_distribution(cos_list[i])
#         print('id:', i)
#         #print('euler:', euler[i])
#         #print('mahat:', mahat[i])
#         #print('qube:', qube[i])
#         #print('cos:', cos[i])
#         result_recording_file.write('image_id:' + str(i))
#         result_recording_file.write('\n')
#         result_recording_file.write('euler:' + str(euler[i]))
#         result_recording_file.write('\n')
#         result_recording_file.write('mahat:' + str(mahat[i]))
#         result_recording_file.write('\n')
#         result_recording_file.write('qube:' + str(qube[i]))
#         result_recording_file.write('\n')
#         result_recording_file.write('cos:' + str(cos[i]))
#         result_recording_file.write('\n')
#         result_recording_file.write('difference:' + str(difference[i]))
#         result_recording_file.write('\n')
#         result_recording_file.write('wnum:' + str(wrong_class_num))
#         result_recording_file.write('\n')
#         result_recording_file.write('num_mc:' + str(max_class_num))
#         result_recording_file.write('\n')
#         result_recording_file.write('fenbu:' + str(cos_dis))
#         result_recording_file.write('\n')
#     result_recording_file.close()