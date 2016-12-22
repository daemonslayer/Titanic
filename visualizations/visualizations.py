#!/usr/bin/env python


from __future__ import print_function
## Remove Warnings
import warnings
warnings.filterwarnings('ignore')
##
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

## Configurations
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
matplotlib.style.use('ggplot')
##

train_data = pd.read_csv('../misc/data/train.csv')
# print(train_data.head())
# print(train_data.describe())
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
print(train_data.describe())
