import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

feature_name = ('surgery','Age','Hospital_Number','rectal_temerature','pulse','respiratory_rate'
                ,'temperature_of_extremities','peripheral_pulse','mucous_membranes','capillary_refill_time'
                ,'pain_level','peristalsis','abdominal_distension','nasogastric_tube','nasogastric_reflux',
                'nasogastric_reflux_PH','feces','abdomen','packed_cell_volume','total_protein','abdominocentesis_appearance',
                'abdomcentesis_total_protein','outcome',"surgical_lesion","lesion_1","lesion_2","lesion_3","cp_data")
train = pd.read_csv("horse-colic_data.csv",sep='\s+',names=feature_name,na_values='?')

print(train['nasogastric_reflux_PH'].dropna())


number_feature = ['rectal_temerature','pulse','respiratory_rate','nasogastric_reflux_PH','packed_cell_volume','total_protein','abdomcentesis_total_protein']
#统计各特征值的各取值频数
for each in feature_name:
    if each not in number_feature:
        print(train[each].value_counts())

#查看各属性值的max,min,mean等
for i in number_feature:
    print(train[i].describe())
   
# print(train.describe())

#画直方图和盒图

def plot_feature(feature):
# for i,each in enumerate(number_feature,1):
    plt.figure(figsize=(10, 15))
    ax1 = plt.subplot(311)
    train[feature].hist()
    plt.title(feature)


    ax2 = plt.subplot(312)
    plt.boxplot(train[feature].fillna(fill_none_mode(feature)))
    plt.title(feature)


    ax3 = plt.subplot(313)
    stats.probplot(train[feature],dist="norm",plot=plt)
    plt.title(feature)
    
    # plt.show()
#缺失值处理
#1.将缺失部分剔除(不太理解)

def delete(train):
    train.dropna(axis=0)
    
    train.to_csv("1.csv")
    
#2.最高频率值来填补缺失值
def fill_none_mode(feature):
    dict_mode = {}
    for num in train.loc[:,feature]:
        if dict_mode.get(num):
            dict_mode[num] +=1
        else:
            dict_mode.setdefault(num,1)
    top_count = 0
    for  key,count in dict_mode.items():
        if key>top_count:
            top_count = key
    train.to_csv("2.csv")

    # train.loc[:,feature]=train.loc[:,feature].fillna(top_count)
    return top_count

#3.通过属性的相关关系来填补缺失值(利用RF填补)
def col_predict(feature):
    feature_notnull = train.loc[train[feature].notnull()]
    feature_isnull  = train.loc[train[feature].isnull()]
    test_x = feature_isnull.drop(feature,axis=1)
    test_x = test_x.fillna(-1)

    X = feature_notnull.drop(feature,axis=1)
    X = X.fillna(-1)
    Y = feature_notnull[feature]

    print(Y.describe())
    print(X.describe())

    rfr = RandomForestRegressor(n_estimators=500)
    rfr.fit(X,Y)
    predict = rfr.predict(test_x)
    train.loc[train[feature].isnull(),feature] = predict
    train.to_csv("3.csv")

#4.通过数据对象之间的相似性来填补缺失值

if __name__ == '__main__':

    for each in number_feature:
        #默认是按照众数填补缺失值
        plot_feature(each)
        col_predict(each)
        plot_feature(each)
        plt.show()

