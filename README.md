# ML_project_template
Template and utils code for machine learning project

## Project Structure
1. src/ all source code of this project 
2. tests/ unit/integration/smoke tests
3. notebooks/ jupyter notebooks 
4. scripts/ python or shell scripts
5. model_finished/ finished model for deployment
6. model_training/ directory to store unfinished model
7. logs/ logs 
8. data/ raw_data, debug_data


## How to run the code

### Step 1: environment setup
```shell script
conda env create -f environment.yaml
conda activate revenue_model

# save conda environment setting
conda env export --no-builds > environment.yaml


# create jupyter kernel
python -m ipykernel install --user --name ml_project --display-name "Python3.8(ml_project)"
```
### Step 2: Model training, eval, or code test

1. Config for training is located in [scripts/train_config.py](scripts/train_config.py)
2. Config for source code is located in [src/config.py](src/config.py)

```shell script
# set env first, or the following code will not work
export PYTHONPATH=./:PYTHONPATH
# split data to train & test data; save data
python scripts/data_cvt.py
# train & save model & eval model
python scripts/model_train.py
# load model & eval model; to specify the model version, refer 
python scripts/model_eval.py 
# code test: add `-s` for more detailed output
pytest -s tests
```
MLflow
```
mlflow run . -e cla_data_cvt

```
## TODO

1. [ ] conda environment setup test
2. [ ] re-organize src/utils
3. [ ] LightGBM support
4. [ ] deep learning support



## Reference 

1. [Cookiecutter Data Science Project](https://drivendata.github.io/cookiecutter-data-science/)
