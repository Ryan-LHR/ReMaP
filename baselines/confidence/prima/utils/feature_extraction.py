import numpy as np
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input
import os
# import keras
import sys
# from datautils import get_data,get_model,data_proprecessing
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

# exp_id = sys.argv[1]
# ptype = sys.argv[2]
# samples = len(get_data(exp_id)[0])
from pathlib import Path
# if __name__ == '__main__'
def feature_extraction(model_name, basedir, m_type):
    """
    使用 accquire_prob 生成的结果进行特征抽取：
    - 读取 predict_probability_vector.npy (ori_prob)
    - 读取 *_<ptype>_prob 目录下的每个 image_id.npy (mutants 的 prob)
    - 计算各类距离特征并写入 <exp_id>_<ptype>_feature.txt
    """
    print(f"Feature Extraction")
    # 确保 basedir 是 Path
    basedir = Path(basedir)
    ptype = m_type
    # 1) 概率总目录：<basedir>/predict_prob
    prob_dir = basedir / "predict_prob"
    prob_dir.mkdir(parents=True, exist_ok=True)

    # 2) 读取 accquire_prob 已经生成的原始预测概率
    predicting_file_path = prob_dir / 'predict_probability_vector.npy'
    if not predicting_file_path.exists():
        raise FileNotFoundError(f"ori_prob file not found: {predicting_file_path}，请先运行 accquire_prob。")

    ori_prob = np.load(predicting_file_path)
    # 样本总数
    num_samples = ori_prob.shape[0]

    # 3) 找到 mutants 概率所在的目录：
    #    accquire_prob 中是：prob_dir / f"{model_name}_{ptype}_prob"
    #    这里通过 ptype 进行匹配（model_name 不从参数传入，就从目录自动推断）
    # global ptype, exp_id  # 保持和原始代码一样依赖全局变量
    prob_path_candidates = list(prob_dir.glob(f"*_{ptype}_prob"))
    if not prob_path_candidates:
        raise FileNotFoundError(
            f"未在 {prob_dir} 下找到 *_${ptype}_prob 目录，请确认 accquire_prob 已运行且 ptype 一致。"
        )
    # 简单起见，取第一个匹配目录
    prob_path = prob_path_candidates[0]

    # 4) 特征输出文件：<predict_prob>/<exp_id>_<ptype>_feature.txt
    feat_dir = basedir / "features"
    file_name = f"{model_name}_{ptype}_feature"
    file_base = feat_dir / file_name
    feature_txt_path = str(file_base) + ".txt"

    feature_dir = Path(feature_txt_path).parent
    feature_dir.mkdir(parents=True, exist_ok=True)

    if Path(feature_txt_path).exists():
        print(f"[SKIP] Features already extracted: {feature_txt_path}")
        return

    # 如果你想每次重跑覆盖旧结果，可以用 'w'；如果想追加，可以用 'a+'
    # 这里保持和原始代码行为一致：在循环中用 'a+' 逐样本写入
    # 先清空一次文件（可选）
    open(feature_txt_path, 'w').close()

    # 5) 逐样本计算特征
    # result = np.argmax(ori_prob, axis=1)  # 如需 predicted label，可以保留，但下方未直接使用
    for i in tqdm(range(num_samples), desc=f"Feature Extraction"):
        a = ori_prob[i]               # 原始输入 i 的概率向量
        max_value = np.max(a)
        max_value_pos = np.argmax(a)

        file_path = prob_path / f"{i}.npy"
        if not file_path.exists():
            # 如果缺某些 image_id 的 mutants prob，可以选择跳过或报错
            # 这里选择跳过并打印提示
            print(f"[WARN] mutants prob file not found for id {i}: {file_path}，跳过该样本。")
            continue

        perturbated_prediction = np.load(file_path)   # 形状：(num_mutants, num_classes)

        # 每个样本 i 的特征累计变量
        euler = 0.0
        mahat = 0.0
        qube = 0.0
        cos = 0.0
        difference = 0.0
        different_class = []
        cos_list = []

        # 遍历该样本的所有 mutants 的概率向量
        for pp in perturbated_prediction:
            pro = pp
            opro = a

            difference += abs(max_value - pp[max_value_pos])
            euler += np.linalg.norm(pro - opro)              # L2
            mahat += np.linalg.norm(pro - opro, ord=1)       # L1
            qube += np.linalg.norm(pro - opro, ord=np.inf)   # L∞

            co = 1 - (np.dot(pro, opro.T) /
                      (np.linalg.norm(pro) * np.linalg.norm(opro)))
            # clip 到 [0,1]
            if co < 0:
                co = 0
            elif co > 1:
                co = 1
            cos += co
            cos_list.append(co)

            if np.argmax(pp) != max_value_pos:
                different_class.append(np.argmax(pp))

        # cos 分布特征（保持原始逻辑）
        cos_dis = cos_distribution(cos_list)

        # 统计类别变化情况
        dic = {}
        for key in different_class:
            dic[key] = dic.get(key, 0) + 1
        wrong_class_num = len(dic)
        if len(dic) > 0:
            max_class_num = max(dic.values())
        else:
            max_class_num = 0

        # print('id:', i)
        # print('euler:', euler)
        # print('mahat:', mahat)
        # print('qube:', qube)
        # print('cos:', cos)
        # print('difference:', difference)
        # print('wnum:', wrong_class_num)
        # print('num_mc:', max_class_num)
        # print('fenbu:', cos_dis)

        # 逐样本写入 txt（保持原代码风格）
        result_recording_file = open(feature_txt_path, 'a+', encoding='utf-8')
        result_recording_file.write('image_id:' + str(i) + '\n')
        result_recording_file.write('euler:' + str(euler) + '\n')
        result_recording_file.write('mahat:' + str(mahat) + '\n')
        result_recording_file.write('qube:' + str(qube) + '\n')
        result_recording_file.write('cos:' + str(cos) + '\n')
        result_recording_file.write('difference:' + str(difference) + '\n')
        result_recording_file.write('wnum:' + str(wrong_class_num) + '\n')
        result_recording_file.write('num_mc:' + str(max_class_num) + '\n')
        result_recording_file.write('fenbu:' + str(cos_dis) + '\n')
        result_recording_file.close()