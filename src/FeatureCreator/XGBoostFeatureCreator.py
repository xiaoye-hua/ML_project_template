# -*- coding: utf-8 -*-
# @File    : XGBoostFeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import numpy as np
from typing import List, Tuple

from src.BaseClass.BaseFeatureCreator import BaseFeatureCreator


class XGBoostFeatureCreator(BaseFeatureCreator):
    def __init__(self, **kwargs) -> None:
        super(XGBoostFeatureCreator, self).__init__()
        self.feature_data = None
        self.feature_cols = None

    def get_seasonality_feature(self):
        pass

    def get_features(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        feature_func = [
            self.get_seasonality_feature
        ]
        for func in feature_func:
            func()
        return self.feature_data, self.feature_cols
