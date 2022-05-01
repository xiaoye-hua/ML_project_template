# -*- coding: utf-8 -*-
# @File    : LogXGBRegressionPipeline.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.compose import TransformedTargetRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import make_column_transformer, make_column_selector
import logging
import pandas as pd
import numpy as np
import os

logging.getLogger(__name__)
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline


class LogXGBRegressionPipeline(XGBRegressionPipeline):
    def train(self, X, y, train_params) -> None:
        train_valid = train_params.get("train_valid", False)
        if train_valid:
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        else:
            train_X, train_y = X.copy(), y.copy()
        pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']
        data_transfomer = make_column_transformer(
            (OrdinalEncoder(), make_column_selector(dtype_include=np.object))
            , (OrdinalEncoder(), make_column_selector(dtype_include=np.bool))
            , remainder='passthrough'
        )
        data_transfomer.fit_transform(df_for_encode_train)
        pipeline_lst.append(
            ('data_transformer', data_transfomer)
        )
        train_X = data_transfomer.transform(train_X)
        logging.info(f"Train data shape after data process: {train_X.shape}")
        grid_search_dict = train_params.get('grid_search_dict', None)
        # if grid_search_dict is not None:
        #     logging.info(f"Grid searching...")
        #     begin = time.time()
        #     self.model_params = self.grid_search_parms(X=train_X, y=train_y, parameters=grid_search_dict)
        #     end = time.time()
        #     logging.info(f"Time consumed: {round((end-begin)/60, 3)}mins")
        # self.xgb = XGBRegressor(**self.model_params)
        self.xgb = TransformedTargetRegressor(
            regressor=XGBRegressor(**self.model_params), func=np.log1p, inverse_func=np.expm1
        )
        if train_valid:
            eval_X = test_X.copy()
            eval_y = test_y.copy()
            eval_X = data_transfomer.transform(eval_X)
            logging.info(f"Eval data shape after data process: {eval_X.shape}")
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric=['mae']
                         , eval_set=[[train_X, train_y], [eval_X, eval_y]])
            print(f"Model params are {self.xgb.get_params()}")
            # self._plot_eval_result()
        else:
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric=['mae']
                         , eval_set=[[train_X, train_y]])
        pipeline_lst.append(('model', self.xgb))
        self.pipeline = Pipeline(pipeline_lst)
        if train_valid:
            self.eval(X=test_X, y=test_y)
