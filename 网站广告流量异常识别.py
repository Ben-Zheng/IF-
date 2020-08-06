# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:16:50 2020

@author: 86178
"""

'''
数据来源
http://support.google.com/analytics/answer/3437719?hl=zh-Hans&ref_topic=3416089
数据字段说明见网站
由于数据中并未标注那些流量属于异常流量或是作弊流量，所以只能使用非监督分析方法进行异常流量检测。
且由于数据中兼有逻辑变量以及数值变量，无法确定数据分布是否符合高斯分布。
本项目中使用了Isolation Forest进行异常检测，该算法对特征要求很低，所以并没有进行数据审查、特征筛选、标准化等操作。
但是，通过非监督的方法实现的异常检测结果只能用来缩小排查范围，仅仅能为业务提供更加精确的执行目标。
'''

import pandas as pd
import numpy as np

data=pd.read_csv('outlier.txt',sep=',')#读取数据
data.isnull().all()#查看缺失值

'bounces和social_socialInteractions两列均是NA，选择丢弃'
data_dropna=data.drop(['bounces','social_socialInteractions'],axis=1)
data_dropna=data_dropna.drop(['clientId'],axis=1)#丢弃客户ID
sam_na=data_dropna.columns[data_dropna.isnull().any()]#筛选出仍然有缺失值维度
'''计算缺失值的样本数量，发现newVisits, pageviews, isVideoAd, isTrueDirect四个维度中，结果依次为
4692
3
10370
5165
其中pageview缺失样本较少可以直接丢弃，由于newVisits是数值类型，1即新访客，nan即非新访客
故用0填充。同样的isVideoAd和isTrueDirect默认为True，故缺失值填充False'''
for i in sam_na:
    print(sum(data_dropna[i].isnull()))
    
data_dropna=data_dropna[data_dropna['pageviews'].notnull()]
data_dropna=data_dropna.reset_index(drop=False)
fill_rule={'newVisits':0,'isVideoAd':False,'isTrueDirect':False}
data_fillna=data_dropna.fillna(fill_rule)

'''拆分数值特征与逻辑特征,其中string_data是逻辑特征，num_data是数值特征'''
str_or_num=(data_fillna.dtypes=='object')#判断每列是否为布尔结果
str_cols=[str_or_num.index[ind] for ind,na_result in enumerate(str_or_num) if na_result==True]#筛选出bool特征
string_data=data_fillna[str_cols]
num_data=data_fillna[[i for i in str_or_num.index if i not in str_cols]]

'''使用OrdinalEncoding将分类特征转化为数值索引'''
from sklearn.preprocessing import OrdinalEncoder
model_oe=OrdinalEncoder()
string_data_con=model_oe.fit_transform(string_data)
string_data_pd=pd.DataFrame(string_data_con,columns=string_data.columns)#将数据格式由array转为DataFrame
'''将两个数据整合在一起'''
featrue_merge=pd.concat([num_data,string_data_pd],axis=1)

'''在本项目中使用IsolationForest进行异常检测'''
from sklearn.ensemble import IsolationForest
model_isof=IsolationForest(n_estimators=20,n_jobs=1)
outlier_label=model_isof.fit_predict(featrue_merge)

'''汇总结果，其中-1为异常，1代表正常,即结果中有两成多是异常值'''
outlier_pd=pd.DataFrame(outlier_label,columns=['outlier_label'])
data_merge=pd.concat((data_fillna,outlier_pd),axis=1)
outlier_count=data_merge.groupby(['outlier_label'])['visitNumber'].count()
print('outliers:{0}/{1}'.format(outlier_count.iloc[0],data_merge.shape[0]))

'''按照来源source整理数据'''
def cal_sample(df):
    data_count=df.groupby(['source'],as_index=False)['outlier_label'].count()
    return data_count.sort_values(['outlier_label'],ascending=False)

'''统计每个渠道的异常情况'''

'''取出异常样本'''
outlier_source=data_merge[data_merge['outlier_label']==-1]
outlier_source_sort=cal_sample(outlier_source)

'''取出正常样本'''
normal_source=data_merge[data_merge['outlier_label']==1]
normal_source_sort=cal_sample(normal_source)

'''合并总样本'''
source_merge=pd.merge(outlier_source_sort,normal_source_sort,on='source')
source_merge=source_merge.rename(index=str,columns={'outlier_label_x':'outlier_count','outlier_label_y':'normal_count'})
source_merge=source_merge.fillna(0)

'''计算异常值的比例'''
source_merge['total_count']=source_merge['outlier_count']+source_merge['normal_count']
source_merge['outlier_rate']=source_merge['outlier_count']/(source_merge['total_count'])
print(source_merge.sort_values(['total_count'],ascending=False))

'''将分析结果可视化'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
#三维可视化视图，使用异常个数，总数，以及异常比例作为三维图的三个方向坐标
ax.scatter(source_merge['outlier_count'],source_merge['total_count'],source_merge['outlier_rate'],s=100,edgecolors='k',c='r',marker='o',alpha=0.5)
ax.set_xlabel('outlier_count')
ax.set_ylabel('total_count')
ax.set_zlabel('outlier_rate')
plt.title('outlier point distribution')





















