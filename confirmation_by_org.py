import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.font_manager import FontProperties


df = pd.read_pickle('../data/pickle/everbridge_data_raw.pickle')

organizations = pd.unique(df['organizationId'])

N = len(organizations)
fig, ax = plt.subplots()
ind = np.arange(N)
width = 0.2
confirmation = []
late_confirmation = []
non_confirmation = []
confirmation_rate = []
for org in organizations:
    df_tmp = df[df.organizationId == org]
    confirm = len(df_tmp[df_tmp.attemptState == 1])
    confirm_late = len(df_tmp[df_tmp.attemptState == 2])
    confirmation.append(confirm)
    late_confirmation.append(confirm_late)
    not_confirm = df_tmp.shape[0] - confirm - confirm_late
    non_confirmation.append(not_confirm)
    confirmation_rate.append((confirm + confirm_late)/(confirm + not_confirm + confirm_late))

print('confirmation rate (including late confirmation): ', confirmation_rate)
rects1 = ax.bar(ind, confirmation, width, color=(199/255, 237/255, 233/255), data=confirmation)
rects2 = ax.bar(ind + width, late_confirmation, width, color=(175/255, 215/255, 237/255), data=late_confirmation)
rects3 = ax.bar(ind + 2 * width, non_confirmation, width, color=(92/255, 167/255, 186/255), data=non_confirmation)
ax.legend((rects1[0], rects2[0], rects3[0]), ('Confirmed', 'Confirmed Late', 'Not Confirmed'))
ax.set_ylabel('confirmation rate')
ax.set_title('confirmation rate by organization')
ax.set_xticks(ind + 1.5 * width)
ax.set_xticklabels(('8928**6', '1332**', '8928**2'))
plt.ylim((0, 24000))
plt.savefig('../pics/confirmation_org_bar.png')