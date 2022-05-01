# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging
import pickle

from scripts.train_config import train_config_detail, train_end_date, train_begin_date, eval_end_date, eval_begin_date, target_threshold
from scripts.train_config import raw_data_path, dir_mark, debug, model_dir
    #debug_data_path, raw_data_path,
# from src.config import session_file_name, hotel_file_name
# =============== Config ============

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


# target_col = train_config_detail[dir_mark]['target_col']
pipeline_class = train_config_detail[dir_mark]['pipeline_class']
feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params = {}
# grid_search_dict = train_config_detail[dir_mark].get('grid_search_dict', None)
# model_params = train_config_detail[dir_mark].get('model_params', {})
train_valid = train_config_detail[dir_mark].get('train_valid', False)
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)

target_col = train_config_detail[dir_mark].get('target_col', reg_target_col)
feature_used = dense_features + sparse_features
# assert feature_used is not None
if not train_config_detail[dir_mark].get('data_dir_mark', False):
    target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
else:
    target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))
logging.info(f"Reading data from {target_raw_data_dir}")
# train_df = pd.read_csv(os.path.join(target_raw_data_dir, 'train.csv'))
eval_df = pd.read_csv(os.path.join(target_raw_data_dir, 'eval.csv'))
test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))


model_path = os.path.join(model_dir, dir_mark)

if feature_clean_func is not None:
    # train_df = feature_clean_func(df=train_df)
    eval_df = feature_clean_func(df=eval_df)
    test_df = feature_clean_func(df=test_df)

print(f"training date: {train_begin_date}, {train_end_date}")
print(f"eval date: {eval_begin_date}, {eval_end_date}")



feature_cols = eval_feature_cols = feature_used
assert set(feature_cols)==set(eval_feature_cols), f"Diff: {set(feature_cols)-set(eval_feature_cols)}"


logging.info(f"Reading features...")
if debug:
    model_params = {
        'n_estimators': 10
    }
else:
    model_params = {
    }


grid_search_dict = {

    'learning_rate': [0.3, 0.1, 0.5]
    , 'n_estimators': [100, 200, 300]
    , 'max_depth': [5, 6, 8]
    , 'min_child_weight': [0.5, 1, 3]
}


train_params = {
    'epoch': 3
    , 'batch_size': 512
    # , "df_for_encode_train": df_for_encode_train[feature_cols]
    , 'train_valid': train_valid
    , 'grid_search_dict': grid_search_dict
}

logging.info(f"Model training...")
cla_model_path = os.path.join(model_dir, 'v1_0328_xgb_cla')
reg_model_path = os.path.join(model_dir, 'v1_0322_xgb')
logging.info(f"Model eval...")
logging.info(f"There are {len(feature_cols)} features:")
logging.info(f"{feature_cols}")