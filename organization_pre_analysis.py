import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.font_manager import FontProperties


df = pd.read_csv('../data/csv/everbridge_data_raw.csv')
organizations = pd.unique(df['organizationId'])
countries = ['GB', 'IN', 'US', 'CA', 'Other']
type = ['Standard', 'Polling']
priority = ['Priority', 'NonPriority']
paths = ['Office Email', 'SMS', 'Home Phone', 'Cell Phone', 'Work Phone', 'Personal Cell Text', 'Other']

country_colors = [(199/255, 237/255, 233/255),
                  (175/255, 215/255, 233/255),
                  (92/255, 167/255, 186/255),
                  (147/255, 224/255, 255/255),
                  (173/255, 195/255, 192/255)]
priority_colors = [(160/255, 191/255, 124/255), (101/255, 147/255, 74/255)]
type_colors = [(252/255, 157/255, 154/255), (249/255, 205/255, 173/255)]
paths_colors = [(153/255, 77/255, 82/255),
                (217/255, 116/255, 43/255),
                (255/255, 150/255, 128/255),
                (230/255, 180/255, 80/255),
                (255/255, 252/255, 153/255),
                (248/255, 237/255, 137/255),
                (232 / 255, 221 / 255, 203 / 255)]

for org in [organizations[1]]:

    df_tmp = df[df.organizationId == org]
    size = len(df_tmp)
    countries_org = [len(df_tmp[df_tmp.country == x]) for x in countries[0:-1]]
    countries_org.append(size - sum(countries_org))
    # print(countries_org)

    paths_org = [len(df_tmp[df_tmp.pathPrompt == x]) for x in paths[0:-1]]
    paths_org.append(size - sum(paths_org))

    type_org = [len(df_tmp[df_tmp.type == 'Standard']), size - len(df_tmp[df_tmp.type == 'Standard'])]

    priority_org = [len(df_tmp[df_tmp.priority == 'Priority']), size - len(df_tmp[df_tmp.priority == 'Priority'])]

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.pie(countries_org, labels=countries, autopct='%1.1f%%', startangle=90, colors=country_colors)
    plt.axis('equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))

    ax = plt.subplot(2, 2, 2)
    plt.pie(type_org, labels=type, autopct='%1.1f%%', startangle=90, colors=type_colors)
    plt.axis('equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))


    ax = plt.subplot(2, 2, 3)
    plt.pie(priority_org, labels=priority, autopct='%1.1f%%', startangle=90, colors=priority_colors)
    plt.axis('equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))

    ax = plt.subplot(2, 2, 4)
    plt.pie(paths_org, labels=paths, autopct='%1.1f%%', startangle=90, colors=paths_colors)
    plt.axis('equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    plt.savefig('../pics/org'+ str(org) +'.png')
    plt.show()
    plt.close()
