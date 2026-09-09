import pandas as pd
import numpy as np
import pickle
# from datautils import get_data,get_model,data_proprecessing
import os
import sys

from baselines.confidence.prima.utils.select_area_perturbated_generator import collect_x_y_from_dataloader


def read_kill_rate_dict(file_name):
    dictfile = open(file_name + '.dict', 'rb')
    kill_rate_file = pickle.load(dictfile)
    if type(kill_rate_file) == dict:
        kill_rate_dict = kill_rate_file
    else:
        kill_rate_dict = {score: letter for score, letter in kill_rate_file}
    return kill_rate_dict

# exp_id = sys.argv[1]
# sample = len(get_data(exp_id)[0])
# ptypes =  sys.argv[2]

# if __name__ == '__main__':
from pathlib import Path
import os
import numpy as np
import pandas as pd

from pathlib import Path
import os
import numpy as np
import pandas as pd

def features_csv_conclusion(model_name, basedir, ptypes, dataloader, is_train):
    """
    在已经完成：
      - perturb_image (生成 mutated_input)
      - accquire_prob (生成 predict_prob / xxx_prob/*.npy, image_perturbation_xxx.dict)
      - feature_extraction (生成 *_feature.txt)
    之后调用，对所有 mutation types 进行汇总，输出一个特征 CSV。

    这里不再使用 get_data 获取 X/Y，而是通过 collect_x_y_from_dataloader(dataloader)
    获取 Y_test 作为标签。
    """

    # ---------- 1. 解析 ptypes，确定 mutation 规则列表 ----------
    domain = ptypes  # input / model / nlp / form

    if domain == 'input':
        input_types = ['gauss', 'reverse', 'black', 'white', 'shuffle']
    elif domain == 'model':
        input_types = ['GF', 'NAI', 'NEB', 'WS']
    elif domain == 'nlp':
        input_types = ['vs', 'vr', 'vrp']
        # 原实现中会把 ptypes 改成 'input'，用于命名前缀
        ptypes = 'input'
    elif domain == 'form':
        input_types = ['ad']
        ptypes = 'input'
    else:
        raise ValueError(f"Unsupported ptypes/domain: {domain}")

    # ---------- 2. 路径与原始预测概率 ----------
    basedir = Path(basedir)
    prob_dir = basedir / "predict_prob"
    feat_dir = basedir / "features"
    predicting_file_path = prob_dir / 'predict_probability_vector.npy'
    if not predicting_file_path.exists():
        raise FileNotFoundError(f"ori_prob not found: {predicting_file_path}，请先运行 accquire_prob。")

    ori_prob = np.load(predicting_file_path)
    num_samples = ori_prob.shape[0]

    # 这里仍然依赖全局的 exp_id 来命名文件，与 feature_extraction 保持一致
    # global exp_id
    # exp_id 例如：'C10-ResNet20'

    all_vectors = []
    all_titles = []

    # ---------- 3. 遍历各个 mutation 类型，读取 feature.txt + kill dict ----------
    for mt in input_types:
        # feature_extraction 生成的文件：
        #   prob_dir / f"{exp_id}_{mt}_feature.txt"
        feature_txt_path = feat_dir / f"{model_name}_{mt}_feature.txt"
        if not feature_txt_path.exists():
            raise FileNotFoundError(f"feature file not found: {feature_txt_path}，请先对 {mt} 运行 feature_extraction。")

        # kill_num 字典，来自 accquire_prob 的 image_perturbation_{mt}.dict 或 model_perturbation_{mt}.dict
        if domain in ['input', 'nlp', 'form']:
            kill_rate_base = prob_dir / f"image_perturbation_{mt}"
        else:  # 'model'
            kill_rate_base = basedir / "mutated_model" / f"model_perturbation_{mt}"

        kill_rate_dict = read_kill_rate_dict(str(kill_rate_base))

        # 读取 feature.txt
        with open(feature_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cos = []
        difference = []
        wrong_class_num = []
        max_class_num = []
        cos_distribution_list = []

        # 解析每一行（保持原逻辑）
        for line in lines:
            if not line.strip():
                continue
            first = line[0]
            if first == 'c':  # cos
                x = float(line[line.find(':') + 1:-1].strip())
                cos.append(x)
            elif first == 'd':  # difference
                x = float(line[line.find(':') + 1:-1].strip())
                difference.append(x)
            elif first == 'n':  # num_mc / max_class_num
                x = int(line[line.find(':') + 1:-1].strip())
                max_class_num.append(x)
            elif first == 'w':  # wnum / wrong_class_num
                x = int(line[line.find(':') + 1:-1].strip())
                wrong_class_num.append(x)
            elif first == 'f':  # fenbu / cos_distribution
                x = eval(line[line.find(':') + 1:-1].strip())
                cos_distribution_list.append(x)

        # 对齐长度简单检查
        if not (len(cos) == len(difference) == len(max_class_num) == len(wrong_class_num)):
            raise ValueError(
                f"Length mismatch in feature file {feature_txt_path}: "
                f"cos={len(cos)}, diff={len(difference)}, "
                f"num_mc={len(max_class_num)}, wnum={len(wrong_class_num)}"
            )

        # kill_num_list：按样本索引排序
        kill_num_list = []
        for i in range(num_samples):
            if i not in kill_rate_dict:
                raise KeyError(f"kill_rate_dict 中缺少样本 {i}，请检查 {kill_rate_base}.dict")
            kill_num_list.append(kill_rate_dict[i])

        # cos_distribution_list: shape ≈ (num_samples, K)，转置为 K 个长度为 num_samples 的 list
        cd = list(np.asarray(cos_distribution_list).T)  # 得到 [cd0, cd1, ..., cd_{K-1}]

        # 当前 mutation type 的所有特征向量（每个长度 = num_samples）
        all_vector = []
        all_vector.append(kill_num_list)
        all_vector.append(cos)
        all_vector.append(difference)
        all_vector.append(max_class_num)
        all_vector.append(wrong_class_num)
        for sub_cd in cd:
            all_vector.append(sub_cd.tolist())

        # 列名
        types_prefix = f"{ptypes}_{mt}_"
        title = [
            types_prefix + 'kill_num',
            types_prefix + 'cos',
            types_prefix + 'difference',
            types_prefix + 'max_class_num',
            types_prefix + 'wrong_class_num',
        ]
        title.extend([types_prefix + f'cos_distribution{i}' for i in range(len(cd))])

        all_titles.extend(title)
        all_vectors.extend(all_vector)

    # ---------- 4. 组装为 DataFrame ----------
    # for v in all_vectors:
    #     print(len(v))  # 保留原 debug

    pd_data_all = pd.DataFrame(np.asarray(all_vectors).T, columns=all_titles)

    # ---------- 5. 从 dataloader 获取 Y_test，并计算 right_or_wrong ----------
    # 不再使用 get_data，直接通过你的工具函数抽取
    X_test, Y_test = collect_x_y_from_dataloader(dataloader)
    Y_test = np.array(Y_test)
    if len(Y_test) != num_samples:
        raise ValueError(f"标签数量 {len(Y_test)} 与 ori_prob 样本数 {num_samples} 不一致，请检查 dataloader 与 accquire_prob 使用的是否同一测试集。")

    y_predict_prob = ori_prob
    y_predict = np.argmax(y_predict_prob, axis=1)

    if is_train:  # 在训练集特征里加入rightness列
        right_or_wrong = []
        for i in range(num_samples):
            if Y_test[i] != y_predict[i]:
                right_or_wrong.append(0)
            else:
                right_or_wrong.append(1)

        rightness_pd = pd.DataFrame(np.array(right_or_wrong), columns=['rightness'])

        # ---------- 6. 拼接并导出 CSV ----------
        result = pd.concat([pd_data_all, rightness_pd], axis=1)
    else:
        result = pd_data_all

    csv_name = f"{ptypes}_{model_name}_feature.csv"
    csv_path = feat_dir / csv_name
    result.to_csv(csv_path, index=False)
    print(f"[INFO] Feature CSV saved to: {csv_path}")
    return csv_path

