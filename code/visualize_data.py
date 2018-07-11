import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
DATA_PATH = '/home/t3min4l/workspace/keras-wine-quality/data'
FIGURE_PATH = '/home/t3min4l/workspace/keras-wine-quality/figures'

def save_fig(fig_id, tight_layout = False, fig_extension='png', resolution=300):
    path = os.path.join(FIGURE_PATH, fig_id + '.' + fig_extension)
    print('Saving figure:', fig_id)
    if  tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)




red_wine = os.path.join(DATA_PATH, 'winequality-red.csv')
white_wine = os.path.join(DATA_PATH, 'winequality-white.csv')


dataFrame_redWine = pd.read_csv(red_wine, sep=';')
dataFrame_whiteWine = pd.read_csv(white_wine, sep=';')

# print('Red Wine')
# print(dataFrame_redWine.describe())
# print('\n\n\n')
# print('White Wine')
# print(dataFrame_whiteWine.describe())
#
# print(dataFrame_redWine.isnull())
# print(dataFrame_whiteWine.isnull())
#
# print(dataFrame_redWine.info())
# print(dataFrame_whiteWine.info())
# print(dataFrame_redWine['alcohol'].value_counts())
# print(dataFrame_whiteWine['alcohol'].value_counts())

# Draw Histogram of Alcohol in % vol
fig, ax = plt.subplots(1, 2)
ax[0].hist(dataFrame_redWine['alcohol'], bins=10, facecolor='red', ec='black', alpha=0.5, label='Red wine') #ec la mau bien cua cot
ax[1].hist(dataFrame_whiteWine['alcohol'], bins=10,facecolor='white', ec='black',alpha=0.5, label='White wine')

fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.5)

ax[0].set_ylim([0,1000])
ax[0].set_xlabel('Alcohol in % vol')
ax[0].set_ylabel('Frequency')
ax[1].set_xlabel('Alcohol in % vol')

fig.suptitle("Distribution of Alcohol in % vol")
save_fig('Distribution of Alcohol in % vol')
plt.show()
plt.close('all')


# Draw Amount of sulphate in each quality cats of Wine
fig2, ax2 = plt.subplots(1, 2, figsize=(8,4))

ax2[0].scatter(dataFrame_redWine['quality'],dataFrame_redWine['sulphates'], color='red', edgecolor='black')
ax2[1].scatter(dataFrame_whiteWine['quality'], dataFrame_whiteWine['sulphates'], color='white', edgecolor='black', lw=0.5)

ax2[0].set_title('Red Wine')
ax2[1].set_title('White Wine')
ax2[0].set_xlabel('Quality')
ax2[1].set_xlabel('Quality')
ax2[0].set_ylabel('Sulphates')
ax2[1].set_ylabel('Sulphates')
ax2[0].set_ylim([0, 2.2])
ax2[1].set_ylim([0, 2.2])
ax2[0].set_xlim([0,10])
ax2[1].set_xlim([0, 10])
fig2.subplots_adjust(bottom = 0.15, hspace=0.1, wspace=0.5)
fig2.suptitle('Wine quality by amount of Sulphate')

save_fig('Wine quality by amount of Sulphate')
plt.show()
plt.close(fig2)

