# -*- coding: utf-8 -*-
# @File    : reg_data_cvt.py
# @Author  : Hua Guo
# @Disc    :

import pandas as pd
import os
import logging

from scripts.train_config import train_config_detail #, train_end_date, train_begin_date, eval_end_date, eval_begin_date, test_begin_date, test_end_date
from scripts.train_config import raw_data_path, dir_mark, debug #, debug_date, offer_served_data_source
from src.utils import check_create_dir
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params = {}
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_used = dense_features + sparse_features
# print(f"training date: {train_begin_date}, {train_end_date}")
# print(f"eval date: {eval_begin_date}, {eval_end_date}")
# print(f"test date: {test_begin_date}, {test_end_date}")

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


model_path = os.path.join('model_training/', dir_mark)

logging.info(f"Reading features...")

# feature_creator =


data, target = fetch_california_housing(as_frame=True, return_X_y=True)
data['MedHouseVal'] = target
train, test = train_test_split(data, test_size=0.15)
train, eval = train_test_split(train, test_size=0.15)
target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
check_create_dir(directory=target_raw_data_dir)



fc = feature_creator_class()
# feature_creator
import time
begin = time.time()
train_df, feature_cols = fc.get_features(df=train)
end = time.time()
print(f"Train data time: {round((end-begin)/3600, 3)} hours")
# if not debug:X
begin = time.time()
eval_df, eval_feature_cols = fc.get_features(df=eval)
end = time.time()
print(f"Eval data time: {round((end-begin)/3600, 3)} hours")
# if not debug:X
begin = time.time()
test_df, test_feature_cols = fc.get_features(df=test)
end = time.time()
print(f"Test data time: {round((end-begin)/3600, 3)} hours")

logging.info(f"Saving data to dir: {target_raw_data_dir}")
train_df.to_csv(os.path.join(target_raw_data_dir, 'train.csv'), index=False)
eval_df.to_csv(os.path.join(target_raw_data_dir, 'eval.csv'), index=False)
test_df.to_csv(os.path.join(target_raw_data_dir, 'test.csv'), index=False)




