import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import datetime
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/csv/everbridge_data_raw.csv')

# train, df = train_test_split(df, test_size = 0.01)

df_1 = df[df.attemptState == 1]
df_1['weekday'] = pd.Series(np.zeros(df_1.shape[0]))

n = 0
for i, row_1 in df_1.iterrows():
    n+=1
    df_1.iloc[i] = datetime.datetime.strptime(row_1.callStartDate, '%Y-%m-%d %H:%M:%S').weekday()
    print(n)

days = pd.unique(df_1['weekday'])
total = np.array([len(df_1[(df_1['weekday'] == x)]) for x in days])
confirm = np.array([len(df_1[(df_1['weekday'] == x) & (df_1['attemptState'] == 1)]) for x in days])

confirm = confirm/total

colors = [(153/255, 77/255, 82/255),
          (217/255, 116/255, 43/255),
          (255/255, 150/255, 128/255),
          (230/255, 180/255, 80/255),
          (255/255, 252/255, 153/255),
          (248/255, 237/255, 137/255),
          (232 / 255, 221 / 255, 203 / 255)]

N = len(confirm)
fig, ax = plt.subplots()
ind = np.arange(N)
width = 0.3
rects1 = ax.bar(ind, confirm[0], width, color=colors[0])
rects2 = ax.bar(ind + width, confirm[1], width, color=colors[1])
rects3 = ax.bar(ind + 2*width, confirm[2], width, color=colors[2])
rects4 = ax.bar(ind + 3*width, confirm[3], width, color=colors[3])
rects5 = ax.bar(ind + 4*width, confirm[4], width, color=colors[4])
rects6 = ax.bar(ind + 5*width, confirm[5], width, color=colors[5])
rects7 = ax.bar(ind + 6*width, confirm[6], width, color=colors[6])
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
          (str(days[0]), str(days[1]), str(days[2]), str(days[3]), str(days[4]), str(days[5]), str(days[6])))
ax.set_ylabel('confirmation rate')
ax.set_title('confirmation rate by weekday')
plt.savefig('../pics/confirmation_weekday_bar.png')







