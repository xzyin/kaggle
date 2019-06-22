#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
获取训练数据的基本信息:
(2) 男女性别比率,以及男女的存活比率,缺失值占比
(5) 配偶/兄弟数目的分布情况
(6) 孩子/父母的分布情况
(7) 乘客票价分布
'''

def survived_information(data_train):
    data_train['Survived'].value_counts().plot(kind='bar')
    plt.ylabel("获救人数", fontproperties=font)
    plt.xlabel("是否获救:0(遇难),1(获救)", fontproperties=font)
    survived_ratio = round(data_train[data_train['Survived'] == 1]['Survived'].values.size
                           / data_train.Survived.size * 100, 2)
    plt.title("所有人员获救情况:获救率:{}%".format(survived_ratio), fontproperties=font)
    plt.show()

def p_class_information(data_train):
    data_train['Pclass'].value_counts().plot(kind='bar')
    plt.ylabel("人数", fontproperties=font)
    plt.xlabel("船船等级:1(头等舱),2(二等舱),3(三等舱)", fontproperties=font)
    first_ratio = round(data_train[data_train['Pclass'] == 1]['Pclass'].values.size
                           / data_train.Pclass.size * 100, 2)
    second_ratio = round(data_train[data_train['Pclass'] == 2]['Pclass'].values.size
                           / data_train.Pclass.size * 100, 2)
    third_ratio = round(data_train[data_train['Pclass'] == 3]['Pclass'].values.size
                           / data_train.Pclass.size * 100, 2)
    plt.title("不同舱位的人数 头等舱:{} 二等舱:{}%, 三等舱:{}%".format(first_ratio, second_ratio, third_ratio), fontproperties=font)
    plt.show()

def age_information(data_train):
    data_train['Age'].plot(kind='kde')
    data_train[data_train['Survived'] == 0]['Age'].plot(kind="kde")
    data_train[data_train["Survived"] == 1]['Age'].plot(kind='kde')
    plt.ylabel("人数", fontproperties=font)
    plt.xlabel("年龄", fontproperties=font)
    plt.title("不同年龄人数的分布", fontproperties=font)
    plt.legend((u"所有人", u"遇难", u"获救"), loc="best")
    plt.show()

def e_marked_information(data_train):
    data_train['Embarked'].value_counts().plot(kind='bar')
    plt.ylabel("人数", fontproperties=font)
    plt.xlabel("码头:S(南安普敦),C(瑟堡),Q(皇后镇)", fontproperties=font)
    s_ratio = round(data_train[data_train['Embarked'] == 'S']['Embarked'].values.size
                           / data_train.Pclass.size * 100, 2)
    c_ratio = round(data_train[data_train['Embarked'] == 'C']['Embarked'].values.size
                           / data_train.Pclass.size * 100, 2)
    q_ratio = round(data_train[data_train['Embarked'] == 'Q']['Embarked'].values.size
                           / data_train.Pclass.size * 100, 2)
    plt.title("不同码头上船人数 S:{} C:{}%, Q:{}%".format(s_ratio, c_ratio, q_ratio), fontproperties=font)
    plt.show()

'''
通过训练集中未缺失的值,训练模型并补充缺失值,同时保存模型
'''
def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:,0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    predictedAges = rfr.predict(unknow_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges.size
    return df, rfr

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

if __name__ == '__main__':
    data_train = pd.read_csv("../data/titanic/train.csv")