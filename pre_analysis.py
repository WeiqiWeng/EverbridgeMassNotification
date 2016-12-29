import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


main_data = pd.read_csv('../data/csv/train_classification_balanced.csv')

labels = ['GB', 'US', 'IN', 'Other']
n_gb = np.sum(main_data.country_GB)
n_us = np.sum(main_data.country_US)
n_in = np.sum(main_data.country_IN)

sizes = [n_gb, n_us, n_in, main_data.shape[0] - n_gb - n_us - n_in]
colors = [(199/255, 237/255, 233/255), (175/255, 215/255, 233/255), (92/255, 167/255, 186/255), (147/255, 224/255, 255/255)]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.axis('equal')
plt.legend(labels)
plt.savefig('../pics/country_pie_chart.png')
plt.close()

labels = ['Personal Cell Text', 'Home Phone', 'Office Email', 'SMS', 'Cell Phone', 'Work Phone', 'Other']
n_1 = np.sum(main_data['pathPrompt_Personal Cell Text'])
n_5 = np.sum(main_data['pathPrompt_Home Phone'])
n_2 = np.sum(main_data['pathPrompt_Office Email'])
n_3 = np.sum(main_data['pathPrompt_SMS'])
n_4 = np.sum(main_data['pathPrompt_Cell Phone'])
n_6 = np.sum(main_data['pathPrompt_Work Phone'])

sizes = [n_1, n_5, n_2, n_3, n_4, n_6, main_data.shape[0] - n_1 - n_2 - n_3 - n_4 - n_5 - n_6]
colors = [(153/255, 77/255, 82/255),
          (217/255, 116/255, 43/255),
          (255/255, 150/255, 128/255),
          (230/255, 180/255, 80/255),
          (255/255, 252/255, 153/255),
          (255/255, 232/255, 130/255),
          (186/255, 40/255, 53/255)]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.axis('equal')
plt.savefig('../pics/path_pie_chart.png')
plt.close()









