# -*- coding: utf-8 -*-
# @File    : XGBRegClaPipeline.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline


class XGBRegClaPipeline(XGBRegressionPipeline):
    def __init__(self, model_path: str, reg_model_path: str, cla_model_path: str, model_training=False, model_params={}, cla_threshold=0.5, **kwargs):
        super(XGBRegClaPipeline, self).__init__(model_path=reg_model_path,  model_training=model_training)
        self.reg_pipeline = self._load_pipeline(model_path=reg_model_path)
        self.cla_pipeline = self._load_pipeline(model_path=cla_model_path)
        self.cla_threshold = cla_threshold
        self.model_path = model_path
        self.eval_result_path = os.path.join(self.model_path, 'eval')

    def _load_pipeline(self, model_path: str) -> Pipeline:
        pipeline = joblib.load(
            filename=os.path.join(model_path, self.model_file_name)
        )
        return pipeline

    def predict(self, X) -> pd.DataFrame:
        # X['cla_prob'] = self.cla_pipeline.predict_proba(X.copy())[:, 1]
        # non_min_df['predict'] = self.reg_pipeline.predict(non_min_df)
        return X['predict']