# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging
import pickle
import mlflow
from datetime import datetime

from scripts.train_config import train_config_detail
from scripts.train_config import raw_data_path, dir_mark, debug, model_dir
from src.config import log_dir

# =============== Config ============
curDT = datetime.now()
date_time = curDT.strftime("%Y%m%d%H")
current_file = os.path.basename(__file__).split('.')[0]
log_file = '_'.join([current_file, date_time, '.log'])
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(log_dir, log_file)
                    )
console = logging.StreamHandler()
logging.getLogger().addHandler(console)


pipeline_class = train_config_detail[dir_mark]['pipeline_class']
feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params_config = train_config_detail[dir_mark].get('model_params', {})
# grid_search_dict = train_config_detail[dir_mark].get('grid_search_dict', None)
# model_params = train_config_detail[dir_mark].get('model_params', {})
train_valid = train_config_detail[dir_mark].get('train_valid', False)
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)
additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

epochs = train_config_detail[dir_mark].get('epochs', None)
batch_size = train_config_detail[dir_mark].get('batch_size', None)
dense_to_sparse = train_config_detail[dir_mark].get('dense_to_sparse', None)
task = train_config_detail[dir_mark].get('task', None) # params for deepFM

target_col = train_config_detail[dir_mark]['target_col']
feature_used = dense_features + sparse_features
# assert feature_used is not None
if not train_config_detail[dir_mark].get('data_dir_mark', False):
    target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
else:
    target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))
logging.info(f"Reading data from {target_raw_data_dir}")


train_df = pd.read_csv(os.path.join(target_raw_data_dir, 'train.csv'))
eval_df = pd.read_csv(os.path.join(target_raw_data_dir, 'eval.csv'))
test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))

print(train_df[feature_used].info())

if feature_clean_func is not None:
    train_df = feature_clean_func(df=train_df)
    eval_df = feature_clean_func(df=eval_df)
    test_df = feature_clean_func(df=test_df)

df_for_encode_train = pd.concat([train_df, eval_df, test_df], axis=0)


feature_cols = eval_feature_cols = feature_used
assert set(feature_cols)==set(eval_feature_cols), f"Diff: {set(feature_cols)-set(eval_feature_cols)}"


model_path = os.path.join(model_dir, dir_mark)


logging.info(f"Reading features...")

if debug:
    model_params = {
        'n_estimators': 10
    }
else:
    model_params = {
        'learning_rate':0.01
        , 'n_estimators': 100
    #     'learning_rate':0.01
    #     , 'n_estimators': 1000
    # , 'max_depth': 8
    #     , 'min_child_weight': 0.5
    #     , 'gamma':1
    #     , 'colsample_bytree': 0.9
    #     , 'subsample': 0.9
    #     , 'reg_alpha': 1

    }

model_params.update(model_params_config)

grid_search_dict = {

    # 'learning_rate': [0.3, 0.1, 0.5]
    # , 'n_estimators': [100, 200, 300]
    # 'min_child_weight': [0.05, 0.5, 1, 10]
    # , 'max_depth': [3, 4, 6, 8]
    # 'gamma': [0.1, 1, 2, 5, 10]
    #  'colsample_bytree': [0.7, 0.8, 0.9]
    # , 'subsample': [0.7, 0.8, 0.9]
    # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

train_params = {
    'epoches': epochs
    , 'batch_size': batch_size
    , 'dense_to_sparse': dense_to_sparse
    , "df_for_encode_train": df_for_encode_train[feature_cols]
    , 'train_valid': train_valid
    , 'eval_X': eval_df[feature_cols]
    , 'eval_y': eval_df[target_col]
    # , 'grid_search_dict': grid_search_dict
    , 'sparse_features': sparse_features
    , 'dense_features': dense_features
}

train_params.update(additional_train_params)

logging.info(f"Model training...")
with mlflow.start_run(run_name='model_train'):
    mlflow.log_params({
        'begin_date': '2022-01-01',
        'end_date': '2022-04-03',
        'train_data_shape': [12000, 343],
        'test_data_shape': [1200, 434]
    })
    pipeline = pipeline_class(model_path=model_path, model_training=True,
                              model_params=model_params, task=task)
    logging.info(f"Origin train data shape : {train_df[feature_cols].shape}")
    pipeline.train(X=train_df[feature_cols], y=train_df[target_col], train_params=train_params)
    logging.info(f"Model saving to {model_path}..")
    pipeline.save_pipeline()

logging.info(f"Loading model from {model_path}")
new_pipeline = pipeline_class(model_path=model_path, model_training=False, model_params={}
                              , load_pmml=True)
logging.info(f"Model eval...")
logging.info(f"There are {len(feature_cols)} features:")
logging.info(f"{feature_cols}")
new_pipeline.eval(X=test_df[feature_cols], y=test_df[target_col],
              performance_result_file='all_data_performance.txt',
              compare_pmml=True)
