import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import os

df_s = []

legend_title = "Activation"

exp_name = "exp21"
legend1 = "tGELU"
path = "./"+exp_name+"/"
list_file = os.listdir(path)

for file in list_file:
    if(file[-3:]=='pkl'):
        path2 = path + file
        file = open(path2, 'rb')        
        temp = pd.DataFrame()
        temp['value'] = pd.Series(pickle.load(file)['test_accuracy'])
        temp['time_step'] = pd.Series(np.arange(1,len(temp['value'])+1))
        temp[legend_title] = pd.Series([legend1]*len(temp['value']))
        df_s.append(temp)
        file.close()


exp_name = "exp32"
legend2 = "GELU"
path = "./"+exp_name+"/"
list_file = os.listdir(path)

for file in list_file:
    if(file[-3:]=='pkl'):
        path2 = path + file
        file = open(path2, 'rb')        
        temp = pd.DataFrame()
        temp['value'] = pd.Series(pickle.load(file)['test_accuracy'])
        temp['time_step'] = pd.Series(np.arange(1,len(temp['value'])+1))
        temp[legend_title] = pd.Series([legend2]*len(temp['value']))
        df_s.append(temp)
        file.close()

dataset = pd.concat(df_s)


sns.lineplot(data = dataset, x ="time_step", y="value", hue =legend_title, estimator = "mean", ci = 'sd')
#plt.title("tGeLU vs GeLU")
plt.xlabel("Evaluation steps ")
plt.ylabel("Test Accuracy")     


