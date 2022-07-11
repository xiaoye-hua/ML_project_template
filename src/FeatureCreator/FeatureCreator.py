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

    def _clean_col_name(self):
        map_dict = {col: col.replace(' ', '_') for col in self.feature_data.columns}
        self.feature_data = self.feature_data.rename(columns=map_dict)

    def add_0(self):
        """
        PMML file debug
        :return:
        """

        length = len(self.feature_data)
        zero_num = int(length*0.5)
        cols = ['zero_0', 'zero_1']
        for col in cols:
            self.feature_data[col] = np.random.random(length) * 20 - 10
            zero_lst = list(set(np.random.choice(list(range(length)), zero_num)))
            self.feature_data[col].iloc[zero_lst] = 0.0

    def get_features(self, df,  **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        self.feature_data = df
        feature_func = [
            self._clean_col_name,
            self.add_0
        ]
        for func in feature_func:
            func()
        return self.feature_data, self.feature_cols
