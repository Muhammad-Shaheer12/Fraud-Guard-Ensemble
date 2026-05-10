from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


class FraudPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()

    @staticmethod
    def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if df[column].dtype.kind in "biufc":
                df[column] = df[column].fillna(df[column].median())
            else:
                mode_series = df[column].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
                df[column] = df[column].fillna(fill_value)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical = [c for c in df.columns if df[c].dtype == "object"]
        if categorical:
            df = pd.get_dummies(df, columns=categorical, drop_first=True)
        return df

    def _split_scale_balance(self, x: pd.DataFrame, y: pd.Series) -> PreparedData:
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=0.30,
            stratify=y,
            random_state=self.random_state,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=self.random_state,
        )

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)
        x_test_scaled = self.scaler.transform(x_test)

        smote = SMOTE(random_state=self.random_state)
        x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

        return PreparedData(
            x_train=x_train_balanced.astype(np.float32),
            y_train=np.asarray(y_train_balanced, dtype=np.float32),
            x_val=x_val_scaled.astype(np.float32),
            y_val=np.asarray(y_val, dtype=np.float32),
            x_test=x_test_scaled.astype(np.float32),
            y_test=np.asarray(y_test, dtype=np.float32),
            feature_names=list(x.columns),
        )

    def prepare_ieee(self, transaction_path: Path, identity_path: Path) -> PreparedData:
        trans_df = pd.read_csv(transaction_path)
        id_df = pd.read_csv(identity_path)

        merged = trans_df.merge(id_df, on="TransactionID", how="left")
        merged = self._fill_missing(merged)

        target = merged["isFraud"].astype(int)
        features = merged.drop(columns=["isFraud"])

        features = self._encode(features)
        return self._split_scale_balance(features, target)

    def prepare_creditcard(self, csv_path: Path) -> PreparedData:
        df = pd.read_csv(csv_path)
        df = self._fill_missing(df)

        target = df["Class"].astype(int)
        features = df.drop(columns=["Class"])
        features = self._encode(features)

        return self._split_scale_balance(features, target)

    def prepare_paysim(self, csv_path: Path) -> PreparedData:
        df = pd.read_csv(csv_path)
        df = self._fill_missing(df)

        target_col = "isFraud" if "isFraud" in df.columns else "isFlaggedFraud"
        target = df[target_col].astype(int)

        drop_cols = [target_col]
        if "nameOrig" in df.columns:
            drop_cols.append("nameOrig")
        if "nameDest" in df.columns:
            drop_cols.append("nameDest")

        features = df.drop(columns=drop_cols)
        features = self._encode(features)

        return self._split_scale_balance(features, target)
