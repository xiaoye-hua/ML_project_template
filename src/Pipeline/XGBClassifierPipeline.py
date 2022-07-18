# -*- coding: utf-8 -*-
# @File    : XGBClassifierPipeline.py
# @Author  : Hua Guo
# @Disc    :
from xgboost.sklearn import XGBClassifier
# from sklearn.pipeline import Pipeline
from sklearn2pmml import PMMLPipeline as Pipeline
from sklearn2pmml import sklearn2pmml
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import joblib
import os
import logging
import mlflow
import mlflow.xgboost
from pypmml import Model

from src.BaseClass.Pipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval


class XGBClassifierPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=True, model_params={}, load_pmml=False, **kwargs) -> None:
        super(XGBClassifierPipeline, self).__init__(model_training=model_training, model_path=model_path)
        self.pipeline = None
        self.pmml_pipeline = None
        if not self.model_training:
            self.load_pipeline(load_pmml=load_pmml)
        self.model_params = model_params
        self.model_file_name = 'pipeline.pkl'
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        self.eval_metric = 'auc'
        self.eval_metric_lst = [self.eval_metric, 'logloss']

    def load_pipeline(self, load_pmml=False) -> None:
        self.pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.model_file_name)
        )
        if load_pmml:
            self.pmml_pipeline = Model.fromFile(os.path.join(self.model_path, 'pipeline.pmml'))

    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        total_len = y.shape[0]
        positve_len = sum(y)
        negative_len = total_len - positve_len
        logging.info(f"positive/negative = {positve_len}/{total_len}={round(positve_len/total_len, 3)}")
        train_valid = train_params.get("train_valid", False)
        if train_valid:
            if train_params.get('eval_X', None) is not None:
                print('Eval data from train_params: ..')
                train_X, train_y = X.copy(), y.copy()
                test_X, test_y = train_params["eval_X"].copy(), train_params['eval_y'].copy()
            else:
                train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        else:
            train_X, train_y = X.copy(), y.copy()
        pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']

        onehot_encode = train_params.get('onehot_encode', [])
        ordinal_encode = train_params.get('ordinal_encode', [])
        passthrough_features = list(set(X.columns)-set(onehot_encode)-set(ordinal_encode))
        transfomer_lst = []
        if onehot_encode != []:
            transfomer_lst.append(
                (self.onehot_encoder_name, OneHotEncoder(handle_unknown='ignore'), onehot_encode)
            )
        if ordinal_encode != []:
            transfomer_lst.append(
                ('ordianl', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-100), ordinal_encode)
            )
        if passthrough_features != []:
            transfomer_lst.append(
                ('passthrough', 'passthrough',passthrough_features)
               )
        data_transfomer = ColumnTransformer(
            transformers=transfomer_lst
        )
        data_transfomer.fit_transform(df_for_encode_train)
        pipeline_lst.append(
            (self.data_transfomer_name, data_transfomer)
        )
        train_X = data_transfomer.transform(train_X)
        logging.info(f"Train data shape after data process: {train_X.shape}")
        grid_search_dict = train_params.get('grid_search_dict', None)
        if grid_search_dict is not None:
            logging.info(f"Grid searching...")
            begin = time.time()
            self.model_params = self.grid_search_parms(X=train_X, y=train_y, parameters=grid_search_dict)
            end = time.time()
            logging.info(f"Time consumed: {round((end-begin)/60, 3)}mins")

        # self.model_params['scale_pos_weight'] = (len(train_y)-sum(train_y))/len(train_y)
        print(self.model_params)
        # mlflow.xgboost.autolog()
        self.xgb = XGBClassifier(**self.model_params)
        print(f"train data feature shape: {train_X.shape}")
        print(f"train data label shape: {train_y.shape}")
        if train_valid:
            eval_X = test_X.copy()
            eval_y = test_y.copy()
            eval_X = data_transfomer.transform(eval_X)
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric=self.eval_metric_lst
                         , eval_set=[[train_X, train_y], [eval_X, eval_y]]
                         ,early_stopping_rounds=10
                         )
            self._plot_eval_result()
        else:
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric=self.eval_metric_lst
                         , eval_set=[[train_X, train_y]]
                         ,early_stopping_rounds=10)
        print(f"Model params are {self.xgb.get_params()}")
        pipeline_lst.append(('model', self.xgb))
        self.pipeline = Pipeline(pipeline_lst)
        if train_valid:
            self.eval(X=test_X, y=test_y, default_fig_dir=os.path.join(self.eval_result_path, 'eval_data'))

    def predict(self, X, pmml=False) -> pd.DataFrame:
        if pmml:
            pmml_res = self.pmml_pipeline.predict(X)
            return pmml_res['probability(1)'].values
        else:
            return self.pipeline.predict_proba(X=X)[:, 1]

    def save_pipeline(self) -> None:
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.model_file_name)
        )[0]
        logging.info(file_name)
        pmml_file = os.path.join(self.model_path, 'pipeline.pmml')
        sklearn2pmml(self.pipeline, pmml_file)
        logging.info(f"Saving pipeline to {pmml_file}")

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True, compare_pmml=False, **kwargs) -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        # transformer = self.pipeline['data_transformer']
        # feature_names = transformer.get_feature_names()
        transfomers = self.pipeline[self.data_transfomer_name].transformers
        feature_cols = []
        for name, encoder, features_lst in transfomers:
            if name == self.onehot_encoder_name and len(features_lst)!=0:
                original_ls = features_lst
                features_lst = self.pipeline[self.data_transfomer_name].named_transformers_[self.onehot_encoder_name].get_feature_names()
                for lst_idx, col in enumerate(features_lst):
                    seg_lst = col.split('_')
                    index = seg_lst[0]
                    cate = '_'.join(seg_lst[1:])
                    index = int(index[1:])
                    original = original_ls[index]
                    features_lst[lst_idx] = '_'.join([cate, original])
            feature_cols += list(features_lst)
        logging.info(f"features num: {len(feature_cols)}")
        logging.info(f"feature_col is {feature_cols}")
        if importance:
            show_feature_num = min(30, len(feature_cols))
            plot_feature_importances(model=self.pipeline['model'],
                                     feature_cols=feature_cols,
                                     show_feature_num=show_feature_num,
                                     fig_dir=fig_dir)
            logging.info(f"Ploting SHAP value:")
            self.plot_shap_importance(X=X, columns_names=feature_cols, fig_dir=fig_dir, show_feature_num=show_feature_num)
        predict_prob = self.pipeline.predict_proba(X=X.copy())[:, 1]
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)
        mlflow.log_metrics({"log_loss": 0.98, "accuracy":0.099})
        if compare_pmml:
            threshold_95 = 0.0001
            X['predict'] = self.predict(X)
            X['pmml_predict'] = self.predict(X, pmml=True)
            X['abs_diff'] = round(abs(X['predict'] - X['pmml_predict']), 2)
            X['abs_diff'].describe([0.05*i for i in range(20)])
            logging.info(X['abs_diff'].describe([0.05*i for i in range(20)]))
            p95 = np.percentile(X['abs_diff'].tolist(), 95)
            assert p95 < threshold_95, f"95 percentile should be smaller than {p95}"

    def plot_shap_importance(self, X, columns_names, show_feature_num, fig_dir=None):
        import shap
        import scipy
        explainer = shap.Explainer(self.pipeline['model'])
        features = self.pipeline[self.data_transfomer_name].transform(X.copy())
        # if isinstance(features, scipy.sparse._csr.csr_matrix):
        #     features = features.toarray()
        df = pd.DataFrame(features, columns=columns_names)
        shap_values = explainer(df)
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=show_feature_num, show=False)
        plt.title("Shap Feature Importance")
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, 'shap_feature_importance.png'), bbox_inches = 'tight')
            # plt.savefig(os.path.join(fig_dir, 'shap_feature_importance.pdf'))
        else:
            plt.show()

        plt.figure()
        shap.plots.bar(shap_values.abs.mean(0), show=False, max_display=show_feature_num)
        plt.title("Shap Feature Importance Bar plot")
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, 'shap_feature_importance_bar.png'), bbox_inches = 'tight')
        else:
            plt.show()

    def _plot_eval_result(self):
        # retrieve performance metrics
        results = self.xgb.evals_result()
        # plot learning curves
        for metric in self.eval_metric_lst:
            plt.plot(results['validation_0'][metric], label='train')
            plt.plot(results['validation_1'][metric], label='test')
            # show the legend
            plt.legend()
            # show the plot
            plt.savefig(os.path.join(self.eval_result_path, f'xgb_train_eval_{metric}.png'))
        # plt.show()

    def grid_search_parms(self, X: pd.DataFrame, y: pd.DataFrame, parameters: dict) -> dict:
        xgb = XGBClassifier()

        xgb_grid = GridSearchCV(xgb,
                                parameters,
                                cv=2,
                                n_jobs=5,
                                verbose=3,
                                 scoring='roc_auc',
                                return_train_score=True
                                )
        xgb_grid.fit(X, y)
        print(xgb_grid.best_score_)
        print(xgb_grid.best_params_)
        return xgb_grid.best_params_