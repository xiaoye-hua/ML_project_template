# -*- coding: utf-8 -*-
# @File    : FeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import numpy as np
from typing import List, Tuple

from src.BaseClass.BaseFeatureCreator import BaseFeatureCreator


class FeatureCreator(BaseFeatureCreator):
    def __init__(self, **kwargs) -> None:
        super(FeatureCreator, self).__init__()
        self.feature_data = None
        self.feature_cols = None

    def get_seasonality_feature(self):
        pass

    def get_features(self, df,  **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        self.feature_data = df
        feature_func = [
            self.get_seasonality_feature
        ]
        for func in feature_func:
            func()
        return self.feature_data, self.feature_cols
