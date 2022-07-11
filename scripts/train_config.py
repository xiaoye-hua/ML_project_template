# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Disc    :
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline
from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.Pipeline.DeepFMPipeline import DeepFMPipeline
from src.Pipeline.XGBClassifierPipeline import XGBClassifierPipeline
from src.Pipeline.XGBRegClaPipeline import XGBRegClaPipeline
from src.Pipeline.XGBoostLR import XGBoostLR
from src.Pipeline.RidgeReg import RidgeReg
from src.config import breast_cancel_traget

from src.config import california_target

debug = False
dir_mark = "iris_cla"
# dir_mark = 'california_deepfm_reg'

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
        , 'target_col': breast_cancel_traget
        # , 'data_dir_mark': 'v1_0501_clareg'
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
        , 'dense_features':
            [
            'zero_0', 'zero_1',
                'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness', 'mean_compactness', 'mean_concavity',
       'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
       'radius_error', 'texture_error', 'perimeter_error', 'area_error',
       'smoothness_error', 'compactness_error', 'concavity_error',
       'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
       'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
       'worst_smoothness', 'worst_compactness', 'worst_concavity',
       'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
        # , 'feature_clean_func': clean_feature
        , 'target_col': breast_cancel_traget
    },
    "iris_deepfm_cla": {
        "pipeline_class": DeepFMPipeline
        , 'task': 'binary'
        , 'feature_creator': FeatureCreator
        , 'epochs': 2
        , 'batch_size': 20
        , 'dense_to_sparse': True
        , 'train_valid': True
        , 'sparse_features': []
        , 'dense_features': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness', 'mean_compactness', 'mean_concavity',
       'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
       'radius_error', 'texture_error', 'perimeter_error', 'area_error',
       'smoothness_error', 'compactness_error', 'concavity_error',
       'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
       'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
       'worst_smoothness', 'worst_compactness', 'worst_concavity',
       'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
        # , 'feature_clean_func': clean_feature
        , 'target_col': breast_cancel_traget
    },
    "california_deepfm_reg": {
        "pipeline_class": DeepFMPipeline
        , 'task': 'regression'
        , 'feature_creator': FeatureCreator
        , 'epochs': 2
        , 'batch_size': 20
        , 'dense_to_sparse': True
        , 'train_valid': True
        , 'sparse_features': []
        , 'dense_features': [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
            'Latitude', 'Longitude'
        ]
        # , 'feature_clean_func': clean_feature
        , 'target_col': california_target
    },

}