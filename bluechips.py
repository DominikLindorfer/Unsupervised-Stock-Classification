#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bluechips.py
@Time    :   2022/01/16 08:29:25
@Author  :   Dominik Lindorfer 
@Contact :   d.lindorfer@3bg.at
@License :   (C)Copyright
@Version :   0.1
@Descr   :   Unsupervised Classification of Stocks based on their Fundamentals
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
files = ["Data_BlueChips/ATX_2014.txt", "Data_BlueChips/DAX_2014.txt", "Data_BlueChips/MDAX_2014.txt", "Data_BlueChips/SPX_2014.txt", "Data_BlueChips/STOXX600_2014.txt", "Data_BlueChips/STOXX50_2014.txt"]
df_mean_all = []

for filename in files:
    print("Processing File: ", filename)

    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = list(map(str.split, lines))

    for i in range(1, len(lines)):
        if lines[i][0] == "ID":
            data.append([lines[i]])
        else:
            if(len(data) > 0):
                data[-1].append(lines[i])

    df_data = []

    columns = ["div_yield", "mkt_cap", "pe_ratio", "pb_ratio", "eps", "debt_assets", "opm"]

    for i in range(len(data)):
        df = pd.DataFrame(data[i][1:])
        df[df.columns[1]] = df[df.columns[1]] + "-" + df[df.columns[2]]
        df.drop(df.columns[[0, 2, 3]], axis = 1, inplace = True)
        df.columns=["TICKER", "DATE", columns[i]]
        
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["DATE_year"] = df["DATE"].apply(lambda x: x.year)
        print(df.columns)
        df.dropna(inplace=True)
        # df.drop(df.columns[1], inplace=True)

        df_data.append(df)

    for d in df_data:
        d.drop(d.columns[[1]], axis = 1, inplace = True)
        print(len(d))

    df_all = reduce(lambda left,right: pd.merge(left,right,how="outer",on=["TICKER", "DATE_year"]), df_data)
    df_all.sort_values(by=["TICKER", "DATE_year"], inplace=True)
    df_all.reset_index(inplace=True, drop=True)

    cols = df_all.columns.drop(["TICKER", "DATE_year"]) 
    df_all[cols] = df_all[cols].apply(pd.to_numeric, errors='coerce')

    # Fill Up Data
    df_all[["div_yield", "eps"]] = df_all[["div_yield", "eps"]].fillna(0)
    df_all["mkt_cap_chg"] = df_all.groupby("TICKER")["mkt_cap"].apply(lambda x: x.diff(periods=1)).fillna(0)
    df_all[["mkt_cap", "debt_assets", "opm"]] = df_all.groupby("TICKER")["mkt_cap", "debt_assets", "opm"].transform(lambda x: x.fillna(x.mean()))

    df_all[["pe_ratio", "pb_ratio"]] = df_all.groupby("TICKER")["pe_ratio", "pb_ratio"].transform(lambda x: x.fillna(x.mean() * 2))

    # Check for NaN Values
    df.isna().sum().sum()

    # Get Mean Values for first approach; Later: Changes in key figures etc.
    df_mean = df_all.groupby("TICKER")["div_yield", "eps", "pe_ratio", "pb_ratio", "mkt_cap", "debt_assets", "opm"].apply(lambda x: x.mean())
    df_mean_all.append(df_mean)

# Concat all Files to one pd Array
df_mean_all = pd.concat(df_mean_all)
df_mean_all = df_mean_all[~df_mean_all.index.duplicated(keep='first')]

df_mean_all[["pe_ratio", "pb_ratio"]] = df_mean_all[["pe_ratio", "pb_ratio"]].apply(lambda x: x.fillna(x.mean()))
df_mean_all["mkt_cap"] = df_mean_all[["mkt_cap"]].apply(lambda x: x.fillna(x.mean()))

# print(df_mean_all.reset_index()[["TICKER", "pb_ratio"]].sort_values(by=["pb_ratio"]).to_string())

"""
Check Data for NaN features
"""
# Display NaNs by feature
nanCounter = np.isnan(df_mean_all).sum()
print(nanCounter)

"""
Do the ML Clustering Analysis
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_train = df_mean_all.drop(columns=["eps", "pb_ratio"]).copy(deep=True)

# Scale data
sX = pp.StandardScaler()
X_train.loc[:,:] = sX.fit_transform(X_train)

# Principal Component Analysis
n_components = 6
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

# Switch PCA on / off
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

# Analyze generated Clusters
clust_1 = X_train_kmeansClustered[X_train_kmeansClustered["cluster"] == 0]
clust_2 = X_train_kmeansClustered[X_train_kmeansClustered["cluster"] == 1]

# Save Result to Pickle
df_mean_all.reset_index(inplace=True)
X_train_kmeansClustered.reset_index(inplace=True)

df_all_bluechips = pd.concat([df_mean_all, X_train_kmeansClustered["cluster"]], axis=1)
pickle.dump(df_all_bluechips, open("result_bluechips.pkl", "wb"))

# Save the properties of sklearn linear scalarer & the kmeans model
std  = np.sqrt(sX.var_)
np.save("result_bluechips_std.npy",std )
np.save("result_bluechips_mean.npy",sX.mean_)
pickle.dump(kmeans, open("kmeansmodel_bluechips.pkl", "wb"))

# load the model
# model = pickle.load(open("kmeansmodel.pkl", "rb"))

# # Predict the class for a test-company
# #Use same scaling from standard scaler as before

# s = np.load('std.npy')
# m = np.load('mean.npy')
# comp_test = [4 for i in range(9)]
# comp_test.append(5)
# comp_test = (np.array(comp_test - m)) / s   # z = (x - u) / s ---> Main formula
# comp_test = pd.DataFrame([comp_test])
# comp_test.columns = X_train.columns
# res = kmeans.predict(comp_test)









# """
# Same Data Pre-Processing as for Divis
# """
# df = df_all
# df["DATE_year"] = df["DATE_year"].apply(lambda x: datetime.datetime(year = x, month=12, day=31))

# # DF for every year until 2012
# y_end = datetime.datetime(year = 2022, month=12, day=31)
# # data_2022 = df[df["DATE"] > y_end.replace(y_end.year - 1)]
# df_years = []

# for year in range(1,9):

#     df_year = df[(df["DATE_year"] > y_end.replace(y_end.year - year - 1)) & (df["DATE_year"] <= y_end.replace(y_end.year - year))]
#     cols = df_year.columns
#     cols = [str(y_end.year - year) + "_" + c if(c != "TICKER") else "TICKER" for c in cols]
    
#     df_year.columns = cols
#     df_years.append(df_year)

# # data_years.insert(0, data_2022)

# for frame in df_years:
#     frame.set_index("TICKER", inplace=True)

# df_years = reduce(lambda left,right: pd.merge(left,right,on='TICKER'), df_years)

# # Multiplier Dataframes - # of divis paid in the last 10 years, mean divi yield, dividend aristocrat?
# multipliers = pd.DataFrame()
# multipliers["TICKER"] = df["TICKER"]
# multipliers["div_chg"] = df.groupby("TICKER")["divs"].apply(lambda x: x.diff(periods=1)).fillna(0)

# df_years["divs_paid"] = df_years.apply(np.count_nonzero, axis=1)
# df_years["div_yield_mean"] = df_years.apply(np.mean, axis=1)

#df_years["divs_paid"] = df_years.groupby("TICKER")["divs"].apply(np.count_nonzero)
#df_years["div_yield_mean"] = df_years.groupby("TICKER")["div_yield"].apply(np.mean)
#df_all["max_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(max)
#df_all["market_cap"] = df.groupby("TICKER")["market_cap"].apply(np.mean)
#div_multipliers2["mean_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(np.mean)
#div_multipliers2["min_div_chg"] = div_multipliers.groupby("TICKER")["div_chg"].apply(min)

