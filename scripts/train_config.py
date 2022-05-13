# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Disc    :
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline
from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.Pipeline.XGBClassifierPipeline import XGBClassifierPipeline
from src.Pipeline.XGBRegClaPipeline import XGBRegClaPipeline
from src.Pipeline.RidgeReg import RidgeReg
from src.config import breast_cancel_traget

from src.config import california_target

debug = False
dir_mark = "iris_cla"

if debug:
    raw_data_path = 'data/debug'
    model_dir = 'model_training/debug'
else:
    raw_data_path = 'data/raw_data'
    model_dir = 'model_training/'




train_config_detail = {
    "v1_0501_xgb_clareg": {
        'cla_dir': 'v1_0501_xgb_cla'
        , 'reg_dir': 'v1_0501_xgb_reg'
        , 'pipeline_class': XGBRegClaPipeline
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'sparse_features': [
        ]
        , 'dense_features': [
        ]
        # , 'feature_clean_func': clean_map_feature
        , 'target_col': ''
        , 'data_dir_mark': 'v1_0501_clareg'
    },

    "california_housing_reg": {
        "pipeline_class": XGBRegressionPipeline
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'sparse_features': []
        , 'dense_features': [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                   'Latitude', 'Longitude'
        ]
        # , 'feature_clean_func': clean_feature
        , 'target_col': california_target
    },
    "iris_cla": {
        "pipeline_class": XGBClassifierPipeline
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'sparse_features': []
        , 'dense_features': [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
                   'mean smoothness', 'mean compactness', 'mean concavity',
                   'mean concave points', 'mean symmetry', 'mean fractal dimension',
                   'radius error', 'texture error', 'perimeter error', 'area error',
                   'smoothness error', 'compactness error', 'concavity error',
                   'concave points error', 'symmetry error', 'fractal dimension error',
                   'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                   'worst smoothness', 'worst compactness', 'worst concavity',
                   'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        # , 'feature_clean_func': clean_feature
        , 'target_col': breast_cancel_traget
    },

}