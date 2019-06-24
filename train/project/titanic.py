#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from matplotlib.font_manager import FontProperties
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn import linear_model
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
    known_age = np.matrix(age_df[age_df.Age.notnull()])
    unknow_age = np.matrix(age_df[age_df.Age.isnull()])
    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, np.ravel(y, order='C'))
    predictedAges = rfr.predict(unknow_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df, rfr

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

def dummies_data(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Emarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dumies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pcalss')
    df = pd.concat([df, dummies_Cabin, dummies_Emarked, dumies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df

def scaler_preprocessing(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(np.array(df['Age']).reshape(-1,1))
    df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1,1), age_scale_param)
    fare_scale_param = scaler.fit(np.array(df['Fare']).reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1), fare_scale_param)
    return (df, age_scale_param, fare_scale_param)

def model(df):
    train_df = df.filter(regex='Survived|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = np.matrix(train_df)
    y = train_np[:, 0]
    X = train_np[:,1:]
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X,np.ravel(y, order='C'))
    return clf

def predict_preprocessing(test_df, age_scaler_param, fare_scaler_param, set_missing_model):
    test_df.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    tmp_df = test_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = np.matrix(tmp_df[test_df['Age'].isnull()])
    x = null_age[:, 1:]
    predictedAges = rfr.predict(x)
    test_df.loc[(test_df.Age.isnull()), 'Age'] = predictedAges
    test_df = set_cabin_type(test_df)
    dummies_df = dummies_data(test_df)
    scaler = preprocessing.StandardScaler()
    dummies_df['Age_scaled'] = scaler.fit_transform(np.array(dummies_df['Age']).reshape(-1,1), age_scaler_param)
    dummies_df['Fare_scaled'] = scaler.fit_transform(np.array(dummies_df['Fare']).reshape(-1,1), fare_scaler_param)
    return dummies_df


if __name__ == '__main__':
    data_train = pd.read_csv("../data/titanic/train.csv")
    data_test = pd.read_csv("../data/titanic/test.csv")
    (df_with_miss_age, rfr) = set_missing_ages(data_train)
    df_with_cabin_type = set_cabin_type(df_with_miss_age)
    dummies_df = dummies_data(df_with_cabin_type)
    (scaled_df, age_scaler_param, fare_scaler_param) = scaler_preprocessing(dummies_df)
    test_df = predict_preprocessing(data_test, age_scaler_param, fare_scaler_param, rfr)
    lr = model(scaled_df)
    test = test_df.filter(regex='SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    print(pd.DataFrame({"columns": list(pd.DataFrame(test).keys()), "coef": list(lr.coef_.T)}))
    predictions = lr.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
    result.to_csv("../data/titanic/submission.csv", index=False)

