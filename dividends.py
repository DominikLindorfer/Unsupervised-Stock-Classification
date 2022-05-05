#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dividends.py
@Time    :   2022/01/16 08:29:25
@Author  :   Dominik Lindorfer 
@Contact :   d.lindorfer@3bg.at
@License :   (C)Copyright
@Version :   0.1
@Descr   :   Unsupervised Classification of Stocks based on their Dividend-Data
'''

import numpy as np
import pandas as pd
import re
import datetime
from sklearn.impute import SimpleImputer
from sklearn import preprocessing as pp
from functools import reduce
import pickle

"""
Pre-Processing Data
"""
files = ["Data_Divis/DAX_Dividend_Data.txt", "Data_Divis/STOXX600_Dividend_Data_2014.txt", "Data_Divis/ATX_Dividend_Data.txt", "Data_Divis/MDAX_Dividend_Data.txt", "Data_Divis/SPX_Dividend_Data.txt"]
df_all = []

for filename in files:

    df = pd.read_table(filename, sep="\s+", header=None, skiprows=1)
    df.fillna('NA', inplace=True)
    df[df.columns[1]] = df[df.columns[1]] + "-" + df[df.columns[2]]
    df.drop(df.columns[[0, 2, 3]], axis = 1, inplace = True)

    with open(filename, 'r') as f:
        header = f.readline().split()

    df.columns = header
    df.rename(columns={"ID" : "TICKER"}, inplace=True)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[["TICKER", "DATE", "divs", "div_yield"]]
    #df.drop(df[["debt_cap", "fcf_yield", "market_cap"]], axis = 1, inplace = True)

    # DF for every year until 2012
    y_end = datetime.datetime(year = 2022, month=12, day=31)
    # data_2022 = df[df["DATE"] > y_end.replace(y_end.year - 1)]
    df_years = []

    for year in range(1,9):

        df_year = df[(df["DATE"] > y_end.replace(y_end.year - year - 1)) & (df["DATE"] <= y_end.replace(y_end.year - year))]
        df_year.drop(columns=["DATE", "divs"], inplace = True)
        cols = df_year.columns
        cols = [str(y_end.year - year) + "_" + c if(c != "TICKER") else "TICKER" for c in cols]
        
        df_year.columns = cols
        df_years.append(df_year)

    # data_years.insert(0, data_2022)

    for frame in df_years:
        frame.set_index("TICKER", inplace=True)

    df_years = reduce(lambda left,right: pd.merge(left,right,on='TICKER'), df_years)

    # Multiplier Dataframes - # of divis paid in the last 10 years, mean divi yield, dividend aristocrat?
    div_multipliers = pd.DataFrame()
    div_multipliers["TICKER"] = df["TICKER"]
    div_multipliers["div_chg"] = df.groupby("TICKER")["divs"].apply(lambda x: x.diff(periods=1)).fillna(0)

    df_years["divs_paid"] = df_years.apply(np.count_nonzero, axis=1)
    df_years["div_yield_mean"] = df_years.apply(np.mean, axis=1)

    #df_years["divs_paid"] = df_years.groupby("TICKER")["divs"].apply(np.count_nonzero)
    #df_years["div_yield_mean"] = df_years.groupby("TICKER")["div_yield"].apply(np.mean)
    #df_all["max_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(max)
    #df_all["market_cap"] = df.groupby("TICKER")["market_cap"].apply(np.mean)
    #div_multipliers2["mean_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(np.mean)
    #div_multipliers2["min_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(min)

    df_all.append(df_years)

df_all = pd.concat(df_all)
df_all = df_all[~df_all.index.duplicated(keep='first')]


# df_all2 = df_all[df_all.index.duplicated(keep='first')]
# doub_df = df_all
# ids = doub_df.index
# doubles = df_all1[~ids.isin(ids[ids.duplicated()])]

"""
Check Data for NaN Values
"""
# Display NaNs by feature
nanCounter = np.isnan(df_all).sum()
print(nanCounter)

"""
Do the ML Clustering Analysis
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_train = df_all.copy(deep=True)

# Scale data
sX = pp.StandardScaler()
X_train.loc[:,:] = sX.fit_transform(X_train)

# Principal Component Analysis
n_components = 8
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA = X_train

# k-means clustering
n_clusters = 2
n_init = 10
max_iter = 300
tol = 0.0001
random_state = 2018

kMeans_inertia = pd.DataFrame(data=[],index=range(2,4), columns=['inertia'])
kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
kmeans.fit(X_train_PCA)
kMeans_inertia.loc[n_clusters] = kmeans.inertia_
X_train_kmeansClustered = kmeans.predict(X_train_PCA)
X_train_kmeansClustered = pd.DataFrame(data = X_train_kmeansClustered, index=X_train_PCA.index, columns=['cluster'])

# print(X_train_kmeansClustered.to_string())
# print(X_train_kmeansClustered.sort_index().to_string())

clust_1 = X_train_kmeansClustered[X_train_kmeansClustered["cluster"] == 0]
clust_2 = X_train_kmeansClustered[X_train_kmeansClustered["cluster"] == 1]

df_all.reset_index(inplace=True)
X_train_kmeansClustered.reset_index(inplace=True)

df_all = pd.concat([df_all, X_train_kmeansClustered["cluster"]], axis=1)
pickle.dump(df_all, open("result_divis.pkl", "wb"))

# Save the properties of sklearn linear scalarer & the kmeans model
std  = np.sqrt(sX.var_)
np.save("result_divis_std.npy",std )
np.save("result_divis_mean.npy",sX.mean_)
pickle.dump(kmeans, open("kmeansmodel_divis.pkl", "wb"))

# load the model
# model = pickle.load(open("kmeansmodel.pkl", "rb"))

# Predict the class for a test-company
#Use same scaling from standard scaler as before

# s = np.load('std.npy')
# m = np.load('mean.npy')
# comp_test = [4 for i in range(9)]
# comp_test.append(5)
# comp_test = (np.array(comp_test - m)) / s   # z = (x - u) / s ---> Main formula
# comp_test = pd.DataFrame([comp_test])
# comp_test.columns = X_train.columns
# res = kmeans.predict(comp_test)






#print(df_all.to_string())

# countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans = analyzeCluster(X_train_kmeansClustered, y_train)


