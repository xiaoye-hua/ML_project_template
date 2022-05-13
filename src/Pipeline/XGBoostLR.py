# -*- coding: utf-8 -*-
# @File    : XGBoostLR.py
# @Author  : Hua Guo
# @Disc    :
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from scipy import sparse
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval

import os
import logging
logging.getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from src.Pipeline.XGBClassifierPipeline import XGBClassifierPipeline
# from src.config import dense_features, criteo_sparse_features


class XGBoostLR(XGBClassifierPipeline):
    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        pipeline_lst = []

        df_for_encode_train = train_params['df_for_encode_train']
        train_valid = train_params.get("train_valid", False)
        dense_features = train_params['dense_features']
        sparse_features = train_params['sparse_features']

        # XGBoost
        self.xgb = XGBClassifier(**self.model_params)
        print(f"Model params are {self.xgb.get_params()}")
        xgb_sparrse_transformer = Pipeline([
            ('ordinal', OrdinalEncoder())
        ])
        self.xgb_transformer = ColumnTransformer(
            transformers=[
                ("xgb_sparse", xgb_sparrse_transformer, sparse_features),
            ]
            , remainder='passthrough'
        )
        self.xgb_transformer.fit(df_for_encode_train.copy())
        logging.info(f"XGB transformer info: ")
        logging.info(self.xgb_transformer)
        if train_valid:
            train_X, test_X, train_y, test_y = train_test_split(self.xgb_transformer.transform(X.copy()), y, test_size=0.2)
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric=self.eval_metric
                         , eval_set=[[train_X, train_y], [test_X, test_y]])
            self._plot_eval_result()
        else:
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric=self.eval_metric
                         , eval_set=[[X, y]])
        # XGB to LR
        leave_info = self.xgb.apply(self.xgb_transformer.transform(X.copy()))#[:, 70:]

        # linear regression
        lr_dense_transformer = Pipeline([
            ('standard', StandardScaler())
            , ('dense_bin', KBinsDiscretizer(n_bins=20, encode='ordinal'))
        ])
        cate_encoder = OneHotEncoder()
        lr_sparse_transformer = Pipeline([
            ('one_hot', cate_encoder)
            , ('pca', TruncatedSVD(n_components=50))
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ("lr_dense", lr_dense_transformer, dense_features),
                ("lr_sparse", lr_sparse_transformer, sparse_features),
            ]
            , remainder='drop'
        )
        self.transformer.fit(df_for_encode_train.copy())
        logging.info(f"LR transformer info: ")
        logging.info(self.transformer)

        logging.info(f"leave dim {X.shape}")
        self.one_hot = OneHotEncoder()
        logging.info(f"one hot...")
        xgb_features = self.one_hot.fit_transform(leave_info)

        logging.info(f"finished one hot")
        lr_feature = self.transformer.transform(X.copy())#self.pipeline.fit_transform(X)
        xgb_features = sparse.csc_matrix(xgb_features)
        all_features = sparse.hstack([xgb_features, lr_feature])
        self.lr = LogisticRegression(
        penalty='l1', solver='saga', verbose=1)
        logging.info(f"Logistic regression training...")
        self.lr.fit(all_features, y)
        logging.info(f"Logistic regression train finished")

    def predict(self, X) -> pd.DataFrame:
        # if self.lr is not None:
        # X = self.ordianl.transform(X)
        lr_feature = self.transformer.transform(X.copy())
        xgb_dense_features = self.xgb_transformer.transform(X.copy())
        leave_info = self.xgb.apply(xgb_dense_features)#[:, 70:]
        xgb_features = self.one_hot.transform(leave_info)
        xgb_features = sparse.csc_matrix(xgb_features)
        all_features = sparse.hstack([xgb_features, lr_feature])
        logging.info(f"finished one hot")
        res = self.lr.predict_proba(X=all_features)[:, 1]
        return res

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True,  **kwargs) -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        # transfomers = self.xgb_transformer.transformers
        # feature_cols = []
        # for name, encoder, features_lst in transfomers:
        #     if name == self.onehot_encoder_name and len(features_lst)!=0:
        #         original_ls = features_lst
        #         features_lst = self.pipeline[self.data_transfomer_name].named_transformers_[self.onehot_encoder_name].get_feature_names()
        #         for lst_idx, col in enumerate(features_lst):
        #             index, cate= col.split('_')
        #             index = int(index[1:])
        #             original = original_ls[index]
        #             features_lst[lst_idx] = '_'.join([cate, original])
        #     feature_cols += list(features_lst)
        # logging.info(f"features num: {len(feature_cols)}")
        # logging.info(f"feature_col is {feature_cols}")
        # if importance:
        #     show_feature_num = min(30, len(feature_cols))
        #     plot_feature_importances(model=self.xgb,
        #                              feature_cols=feature_cols,
        #                              show_feature_num=show_feature_num,
        #                              fig_dir=fig_dir)
        predict_prob = self.predict(X=X.copy())
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)