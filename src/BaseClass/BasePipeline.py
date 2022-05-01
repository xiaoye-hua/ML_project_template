# -*- coding: utf-8 -*-
# @File    : BasePipeline.py
# @Author  : Hua Guo
# @Disc    :
from abc import ABCMeta, abstractmethod
import os
import logging
logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        self.model_training = model_training
        self.model_path = model_path
        self.eval_result_path = os.path.join(self.model_path, 'eval_test')
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        self.model_file_name = 'pipeline.pkl'

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