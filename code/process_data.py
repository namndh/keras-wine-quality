import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

DATA_PATH = '/home/t3min4l/workspace/keras-tutorial/data'

red_wine_data_path = os.path.join(DATA_PATH, 'winequality-red.csv')
white_wine_data_path = os.path.join(DATA_PATH, 'winequality-white.csv')

red_wine = pd.read_csv(red_wine_data_path, sep=';')
white_wine = pd.read_csv(white_wine_data_path, sep=';')

red_wine['type'] = 1
white_wine['type'] = 0

wines = red_wine.append(white_wine, ignore_index=True)
corr = wines.corr()

sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

plt.show()