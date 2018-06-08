# -*- coding: utf-8 -*-
import numpy as np
import datetime as dt
from time import time
import pandas as pd
import numpy as np
import xgboost as xgb

# MPI::Init
xgb.rabit.init()

# Some checks on world size, ranks and hostnames
print('MPI world-size: '+str(xgb.rabit.get_world_size()))
print('MPI get-rank  : '+str(xgb.rabit.get_rank()))
print('MPI hostname  : '+str(xgb.rabit.get_processor_name()))

train_X = 'train_data.txt'

xgdmat = xgb.DMatrix(train_X)                       

our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.85, 'colsample_bytree': 0.85, 'objective': 'multi:softmax', 'num_class': 3, 'max_depth':6, 'min_child_weight':1}

my_model = xgb.train(our_params, xgdmat, num_boost_round = 432)

if xgb.rabit.get_rank() == 0:
    my_model.save_model('saved.model')

# MPI::Finalize
xgb.rabit.finalize()


