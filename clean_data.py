# -- coding: gbk --
import pandas as pd
import time
import re
import numpy as np


csv_path = r'D:\360MoveData\Users\10539\Desktop\data.csv'
clean_path = r'D:\360MoveData\Users\10539\Desktop\data cleaned.csv'

data = pd.read_csv(csv_path, header=0, index_col='id')


data['updated_time'] = data['updated_time'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
data['net_support'] = data['ups'] - data['downs']
data['heat'] = data['ups'] + data['downs']
data['heat'] = (data['heat'] - data['heat'].min()) / (data['heat'].max() - data['heat'].min())
data['score'] = data['stars']*2
data['spent'] = data['spent'].replace(0, np.nan)
data['contents'] = data['contents'].apply(lambda x: re.sub('&[\w]+;', '', str(x)))
data['contents'] = data['contents'].apply(lambda x: re.sub('\(\s*\)', '', str(x)))

data.drop(['ups', 'downs','device'], axis=1, inplace=True)


data.to_csv(clean_path, encoding='utf_8_sig')