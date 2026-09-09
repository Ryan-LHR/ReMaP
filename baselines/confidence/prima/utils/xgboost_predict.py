import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRanker
from sklearn.preprocessing import MinMaxScaler


def rank_prima_tests(
    model: XGBRanker,
    scaler: MinMaxScaler,
    feature_names,
    input_feat_csv,
    model_feat_csv,
    id_col=None,
) -> pd.DataFrame:
    """
    Use a trained PRIMA XGBRanker to rank test inputs.

    Args:
        model:          Trained XGBRanker from `train_prima_xgboost`.
        scaler:         MinMaxScaler fitted on training features.
        feature_names:  Feature column order used in training.
        input_feat_csv: CSV path of input-mutation features for *new* data.
        model_feat_csv: CSV path of model-mutation features for *new* data.
        id_col:         Optional column name used as test case id
                        (e.g. 'sample_id', 'img_idx'). If None, use row index.
        save_rank_path: Optional CSV path to save the ranking result.

    Returns:
        ranking_df: DataFrame with columns:
            - 'rank'         : rank index (1 = highest priority)
            - 'score'        : predicted bug-revealing score
            - 'sample_idx'   : row index in the feature CSVs
            - id_col         : (if provided) original id
            - 'rightness'    : (if存在) 是否被原模型预测正确 (1=正确,0=错误)
    """

    input_feat_csv = Path(input_feat_csv)
    model_feat_csv = Path(model_feat_csv)

    # 1) Load CSVs
    input_df = pd.read_csv(input_feat_csv)
    model_df = pd.read_csv(model_feat_csv)

    # 2) 去掉多余的索引列
    for df in (input_df, model_df):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # 3) 取 rightness（如果有的话，仅用于结果展示，不参与特征）
    rightness = None
    if "rightness" in input_df.columns:
        rightness = input_df["rightness"].values
    elif "rightness" in model_df.columns:
        rightness = model_df["rightness"].values

    # 4) 可选：取一个明确的样本 id 列
    if id_col is not None:
        if id_col not in input_df.columns and id_col not in model_df.columns:
            raise ValueError(f"id_col '{id_col}' not found in either feature CSV.")
        if id_col in input_df.columns:
            ids = input_df[id_col].values
        else:
            ids = model_df[id_col].values
    else:
        ids = np.arange(len(input_df))  # 用行号当 id

    # 5) 拼接特征（按列拼：input 特征 + model 特征），剔除 rightness 和 id 列
    drop_cols = ["rightness"]
    if id_col is not None:
        drop_cols.append(id_col)

    X_input = input_df.drop(columns=[c for c in drop_cols if c in input_df.columns])
    X_model = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

    X_new = pd.concat([X_input, X_model], axis=1)

    # 确保列顺序和训练时一致（非常关键）
    missing_cols = [c for c in feature_names if c not in X_new.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in new data: {missing_cols}")

    X_new = X_new[feature_names]

    # 6) 用训练时的 scaler 做 min-max 归一化
    X_scaled = scaler.transform(X_new)

    # 7) 用 XGBRanker 预测“bug-revealing 分数”
    scores = model.predict(X_scaled)  # shape: (N,)

    # 8) 组装排序结果
    ranking_df = pd.DataFrame({
        "sample_idx": np.arange(len(scores)),
        "score": scores,
    })

    if id_col is not None:
        ranking_df[id_col] = ids

    if rightness is not None:
        ranking_df["rightness"] = rightness

    # 分数越大，优先级越高
    ranking_df.sort_values("score", ascending=False, inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)
    ranking_df.insert(0, "rank", ranking_df.index + 1)  # rank 从 1 开始

    # 9) 可选保存到 CSV
    # if save_rank_path is not None:
    #     save_rank_path = Path(save_rank_path)
    #     save_rank_path.parent.mkdir(parents=True, exist_ok=True)
    #     ranking_df.to_csv(save_rank_path, index=False)

    return ranking_df
