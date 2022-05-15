# -*- coding: utf-8 -*-
# @File    : Pipeline.py
# @Author  : Hua Guo
# @Disc    :
from abc import ABCMeta, abstractmethod
import os
import tensorflow as tf
from typing import Tuple
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
from src.utils.plot_utils import binary_classification_eval, plot_feature_importances
logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        self.model_training = model_training
        self.model_path = model_path
        self.eval_result_path = os.path.join(self.model_path, 'eval_test')
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        self.model_file_name = 'pipeline.pkl'
        self.data_transfomer_name = 'data_transformer'
        self.model_name = 'model'
        self.onehot_encoder_name = 'onehot'

    @abstractmethod
    def train(self, X, y, train_params):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def load_pipeline(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)


# class GBTPipeline(metaclass=BasePipeline):
#     """
#     BasePipeline class for XGBoost ang LightGBM
#     """
#     def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True,  **kwargs) -> None:
#         if default_fig_dir is None:
#             fig_dir = self.eval_result_path
#         else:
#             fig_dir = default_fig_dir
#         self._check_dir(fig_dir)
#         # transformer = self.pipeline['data_transformer']
#         # feature_names = transformer.get_feature_names()
#         transfomers = self.pipeline[self.data_transfomer_name].transformers
#         feature_cols = []
#         for name, encoder, features_lst in transfomers:
#             if name == self.onehot_encoder_name:
#                 original_ls = features_lst
#                 features_lst = self.pipeline[self.data_transfomer_name].named_transformers_[self.onehot_encoder_name].get_feature_names()
#                 for lst_idx, col in enumerate(features_lst):
#                     index, cate= col.split('_')
#                     index = int(index[1:])
#                     original = original_ls[index]
#                     features_lst[lst_idx] = '_'.join([cate, original])
#             feature_cols += list(features_lst)
#         logging.info(f"features num: {len(feature_cols)}")
#         logging.info(f"feature_col is {feature_cols}")
#         if importance:
#             show_feature_num = min(30, len(X.columns))
#             plot_feature_importances(model=self.pipeline['model'],
#                                      feature_cols=feature_cols,
#                                      show_feature_num=show_feature_num,
#                                      fig_dir=fig_dir)
#         predict_prob = self.pipeline.predict_proba(X=X.copy())[:, 1]
#         binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)


class BaseDNNPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(BaseDNNPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        if model_training:
            self.pipeline = None
            self.model = None
        else:
            self.pipeline, self.model = self.load_pipeline()

    def predict(self, X):
        X = self.pipeline.transform(X)
        return self.model.predict(X)

    def load_pipeline(self) -> Tuple[Pipeline, tf.keras.models.Model]:
        pipeline = tf.keras.models.load_model(self.model_path)
        pre_pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )
        return pre_pipeline, pipeline

    def save_pipeline(self) -> None:
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )[0]
        tf.keras.models.save_model(model=self.model, filepath=self.model_path)
        logging.info(f'Model saved in {self.model_path}')

    def train(self, X, y, train_params):
        pass