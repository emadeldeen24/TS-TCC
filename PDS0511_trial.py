# Databricks notebook source
import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
import torch.nn.functional as F
# Args selections
start_time = datetime.now()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip freeze

# COMMAND ----------

# MAGIC %md # 0. Params

# COMMAND ----------

device = torch.device('cpu')
experiment_description = 'Exp1'
data_type = 'Epilepsy' 
method = 'TS-TCC'
training_mode = 'self_supervised'
run_description = 'run1' 

logs_save_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0511_TSTCC/trail/experiments_logs'
seed = 0

# COMMAND ----------

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# COMMAND ----------

SEED = seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# COMMAND ----------

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# COMMAND ----------

# loop through domains
counter = 0
src_counter = 0

# COMMAND ----------

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# COMMAND ----------

# MAGIC %md # 1. Load data

# COMMAND ----------

# MAGIC %md ## step 0: raw data 

# COMMAND ----------

import pandas as pd

# COMMAND ----------

data = pd.read_csv('/Workspace/Repos/fguo@cmtelematics.com/TS-TCC/data_preprocessing/epilepsy/data_files/data.csv')
data.head()

# COMMAND ----------

# MAGIC %md **Note:** 
# MAGIC - Even through y has 5 values, "All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure."
# MAGIC - Essentially, this is a binary classification problem. 

# COMMAND ----------

# MAGIC %md ## step 1: torch data 

# COMMAND ----------

from dataloader.augmentations import DataTransform

# COMMAND ----------

data_path = f"./data/{data_type}"
train_dataset = torch.load(os.path.join(data_path, "train.pt"))

# COMMAND ----------

train_dataset['samples'].shape, train_dataset['labels'].shape

# COMMAND ----------

X_train = train_dataset["samples"]
y_train = train_dataset["labels"]

aug1, aug2 = DataTransform(X_train, configs)

# COMMAND ----------

X_train[0, 0, :]

# COMMAND ----------

aug1[0, 0, :]

# COMMAND ----------

aug2[0, 0, :]

# COMMAND ----------

configs.batch_size, configs.num_epoch

# COMMAND ----------

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# COMMAND ----------

# MAGIC %md # 2. Model training

# COMMAND ----------

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

# COMMAND ----------

model

# COMMAND ----------

temporal_contr_model

# COMMAND ----------

model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# COMMAND ----------

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# COMMAND ----------

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# COMMAND ----------

Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

# COMMAND ----------

model.eval()
temporal_contr_model.eval()

# COMMAND ----------

X_train[:1, :1, :]

# COMMAND ----------

predictions, features = model(X_train[:2, :1, :].float())
features

# COMMAND ----------

out = F.max_pool1d(features, kernel_size = features.size(2),).transpose(1, 2)
out = out.squeeze(1)
out.shape

# COMMAND ----------

features.size(2)

# COMMAND ----------

out

# COMMAND ----------


