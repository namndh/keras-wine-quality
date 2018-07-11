import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

DATA_PATH = '/home/t3min4l/workspace/keras-wine-quality/data'
FIGURE_PATH = '/home/t3min4l/workspace/keras-wine-quality/figures'

def save_fig(fig_id, tight_layout = False, fig_extension='png', resolution=800):
    path = os.path.join(FIGURE_PATH, fig_id + '.' + fig_extension)
    print('Saving figure:', fig_id)
    if  tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



red_wine_data_path = os.path.join(DATA_PATH, 'winequality-red.csv')
white_wine_data_path = os.path.join(DATA_PATH, 'winequality-white.csv')

red_wine = pd.read_csv(red_wine_data_path, sep=';')
white_wine = pd.read_csv(white_wine_data_path, sep=';')

red_wine['type'] = 1
white_wine['type'] = 0

wines = red_wine.append(white_wine, ignore_index=True)
corr = wines.corr()

# heatmap_plot = sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
# save_fig('Correlation of constants in data-set', tight_layout=True)
# plt.show(

# print(wines.head())
wines_copy = wines.copy()
wines_copy = wines_copy.drop(['citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'alcohol', 'quality'], axis=1)
wines_plot = wines_copy.drop(['type'], axis=1)
# print(wines.head())
# print(wines_copy.head())



scaler = StandardScaler()

wines_scaled = scaler.fit_transform(wines_plot)
wines_scaled = pd.DataFrame(wines_scaled, columns=['fixed acidity', 'volatile acidity', 'chlorides', 'density', 'pH', 'sulphates'])

X = wines_scaled.ix[:,0:6]
X = np.matrix(X.as_matrix())
print(type(X))
y = np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('{}.{}.{}.{}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(wines_plot['fixed acidity'], ax=ax1)
# sns.kdeplot(wines_plot['volatile acidity'], ax=ax1)
# sns.kdeplot(wines_plot['chlorides'], ax=ax1)
# sns.kdeplot(wines_plot['density'], ax=ax1)
# sns.kdeplot(wines_plot['pH'], ax=ax1)
# sns.kdeplot(wines_plot['sulphates'], ax=ax1)
# ax2.set_title('After Scaling')
# sns.kdeplot(wines_scaled['fixed acidity'], ax=ax2)
# sns.kdeplot(wines_scaled['volatile acidity'], ax=ax2)
# sns.kdeplot(wines_scaled['chlorides'], ax=ax2)
# sns.kdeplot(wines_scaled['density'], ax=ax2)
# sns.kdeplot(wines_scaled['pH'], ax=ax2)
# sns.kdeplot(wines_scaled['sulphates'], ax=ax2)
# save_fig('Applied StandScaler to dataset', tight_layout=True)
# plt.show()