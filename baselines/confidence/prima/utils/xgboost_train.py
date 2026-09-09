import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRanker


def train_prima_xgboost(
    input_feat_csv: str,
    model_feat_csv: str,
    save_model_path=None,
):
    """
    Train an XGBoost learning-to-rank model for PRIMA.

    Args:
        input_feat_csv:  CSV path of input-mutation features
                        (features_csv_conclusion 生成的 input_feature.csv).
        model_feat_csv:  CSV path of model-mutation features
                        (features_csv_conclusion 生成的 model_feature.csv).
        save_model_path: Optional. If given, save the trained XGBoost model
                         to this path via `model.save_model()`.

    Returns:
        model:  trained XGBRanker model.
        scaler: fitted MinMaxScaler used for feature normalization.
        feature_names: list of feature column names (after concatenation).
    """
    print(f"\nLearning to rank")
    input_feat_csv = Path(input_feat_csv)
    model_feat_csv = Path(model_feat_csv)

    # 1) Load CSVs
    input_df = pd.read_csv(input_feat_csv)
    model_df = pd.read_csv(model_feat_csv)

    # 2) Drop possible index columns like 'Unnamed: 0'
    for df in (input_df, model_df):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # 3) Extract labels according to "whether the input is incorrectly predicted"
    #    features_csv_conclusion 里通常有一列 'rightness':
    #       rightness = 1 -> correctly predicted
    #       rightness = 0 -> incorrectly predicted
    if "rightness" not in input_df.columns:
        raise ValueError("Column 'rightness' not found in input feature CSV.")

    # Make sure both CSVs agree on rightness
    if "rightness" in model_df.columns:
        if not np.array_equal(input_df["rightness"].values, model_df["rightness"].values):
            raise ValueError("Mismatch in 'rightness' column between input and model feature CSVs.")

    # Label y: 1 for incorrectly predicted, 0 for correctly predicted
    y = (input_df["rightness"] == 0).astype(int).values  # shape: (N,)

    # 4) Build feature matrix X by concatenating input & model features
    X_input = input_df.drop(columns=["rightness"])
    X_model = model_df.drop(columns=["rightness"])
    X = pd.concat([X_input, X_model], axis=1)

    feature_names = list(X.columns)

    # 5) Min-max normalization to [0, 1] for each feature (per paper)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6) Prepare group information for learning-to-rank
    #    Here we treat all instances as a single query / list to be ranked.
    group = [len(y)]

    # 7) Define and train XGBoost ranking model
    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_scaled,
        y,
        group=group,
    )

    # 8) Optionally save the trained model
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_model_path))

    return model, scaler, feature_names
