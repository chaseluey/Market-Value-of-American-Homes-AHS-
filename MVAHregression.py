# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:21:48 2022

@author: chase
"""

#keepList = ['HINCP','MARKETVAL','MORTAMT','BEDROOMS','BATHROOMS','TOTROOMS','PORCH',
#            'FIREPLACE','LOTSIZE','UNITSIZE','OWNLOT','GARAGE','DINING','RATINGNH','NHQSCHOOL',
#            'NHQSCRIME','NHQPCRIME','YRBUILT','MAINTAMT','NEARWATER','WATFRONT','NEARTRASH','GATED',
#                'YEARBUY','HOWBUY','HHRACE','HHAGE','ADEQUACY']

import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns; sns.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# READ IN
filepath = "C://Users\chase\Dropbox\matthew-luey\project\stage-2\AmericanHousingSurvey\AHS_PUF_Flat/"

df = pd.read_stata(filepath + "ahs2019n_age_20_64.dta")

#LIST OF VARIABLES TO KEEP
keepList = ['HINCP','MARKETVAL','MORTAMT','BEDROOMS','BATHROOMS','TOTROOMS','PORCH',
            'FIREPLACE','LOTSIZE','UNITSIZE','OWNLOT','GARAGE','DINING','RATINGNH','NHQSCHOOL',
            'NHQSCRIME','NHQPCRIME','YRBUILT','MAINTAMT','NEARTRASH',
                'HOWBUY','HHRACE','HHAGE','ADEQUACY']

#KEEPS ONLY CHOSEN VARIABLES
df_temp = df[keepList]

# SAVE SO I CAN VIEW IN EXCEL
df_temp.to_excel("C://Users\chase\Dropbox\matthew-luey\project\stage-2\AmericanHousingSurvey\AHS_PUF_Flat\DESTRING/" + 'keptVariables.xlsx', index=False)

# # READ IN
filepath = "C://Users\chase\Dropbox\matthew-luey\project\stage-2\AmericanHousingSurvey\AHS_PUF_Flat/"

df_old = pd.read_stata(filepath + "ahs2019n_age_65up.dta")

#KEEPS ONLY CHOSEN VARIABLES
df_old = df[keepList]

# SAVE SO I CAN VIEW IN EXCEL
df_old.to_excel("C://Users\chase\Dropbox\matthew-luey\project\stage-2\AmericanHousingSurvey\AHS_PUF_Flat\DESTRING/" + 'keptVariables.xlsx', index=False)

#SAVE DATAFRAME
df_temp.to_pickle('temp_data')
df = pd.read_pickle('temp_data')

#SAVE DATAFRAME
df_old.to_pickle('temp_data_65up')
df_old = pd.read_pickle('temp_data_65up')

df = df.append(df_old)

numericList = ['BATHROOMS', 'LOTSIZE', 'UNITSIZE', 'PORCH', 'OWNLOT', 'GARAGE', \
               'NHQSCHOOL', 'NHQSCRIME', 'NHQPCRIME','HOWBUY','NEARTRASH']

for i_var in numericList:
    print("Convert " + i_var + " to numeric ...")
    df[i_var] = pd.to_numeric(df[i_var], errors = 'coerce')

df_temp = df

df_temp['MORTAMT'].describe()
df_temp = df[df['MORTAMT'] < 3260]


df_temp['HINCP'].describe()
df_temp = df[(df['HINCP'] < 272862.5)&(df['MORTAMT'] < 3260)&(df['MORTAMT']>0)&(df['MARKETVAL'] < 824060.0)]

#HISTOGRAM FOR INCOME
B = 100
fig, ax = plt.subplots()
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
ax.set_title('INCOME')
prob, bins, patches = ax.hist(df_temp['HINCP'], bins=B, align='mid')

# #REMOVES OUTLIERS
# print(df_temp['MARKETVAL'].describe())
# print(df_temp['MARKETVAL'].hist(bins=60))
# df_temp = df_temp[df_temp['MARKETVAL'] < 824060.0]

df_temp['RATIO'] = df_temp['MARKETVAL']/df_temp['MORTAMT']

#HISTOGRAM FOR MARKET VALUE
B = 100
fig, ax = plt.subplots()
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
ax.set_title('MARKET VALUE')
prob, bins, patches = ax.hist(df_temp['MARKETVAL'], bins=B, align='mid')

#BOXPLOT FOR MARKET VALUE
fig, ax = plt.subplots()
ax.boxplot(df_temp['MARKETVAL'])
ax.set_title('Boxplot of Market Value')
plt.show()

# df_temp = df

#DROP ALL NaN
# df_temp = df_temp.dropna()

#CHECKING FOR CATEGORICAL VARIABLES AND CREATING DUMMIES
# print(df_temp['PORCH'].hist())
df_temp['d_PORCH'] = 0
df_temp.loc[(df_temp['PORCH'] == 1), 'd_PORCH'] = 1

# print(df_temp['GARAGE'].hist())
df_temp['d_GARAGE'] = 0
df_temp.loc[(df_temp['GARAGE'] == 1), 'd_GARAGE'] = 1
print(df_temp['GARAGE'] == 1)

# print(df_temp['OWNLOT'].hist())
df_temp['d_OWNLOT'] = 0
df_temp.loc[(df_temp['OWNLOT'] == 1), 'd_OWNLOT'] = 1

# print(df_temp['DINING'].hist())
# df_temp['d_DININGmissing'] = 0
# df_temp['d_DININGyes'] = 0
# df_temp['d_DININGno'] = 0
# df_temp.loc[(df_temp['DINING'] == 0), 'd_DININGmissing'] = 1
# df_temp.loc[(df_temp['DINING'] == 1), 'd_DININGyes'] = 1
# df_temp.loc[(df_temp['DINING'] == 2), 'd_DININGno'] = 1

# print(df_temp['NHQSCHOOL'].hist())
df_temp['d_NHQSCHOOL'] = 0
df_temp.loc[(df_temp['NHQSCHOOL'] == 1), 'd_NHQSCHOOL'] = 1

# print(df_temp['UNITSIZE'].hist())

# print(df_temp['RATINGNH'].hist())
# df_temp['d_RATINGNHeauals1'] = 0
# df_temp['d_RATINGNHequals2'] = 0
# df_temp['d_RATINGNHequals3'] = 0
# df_temp['d_RATINGNHequals4'] = 0
# df_temp['d_RATINGNHequals5'] = 0
# df_temp['d_RATINGNHequals6'] = 0
# df_temp['d_RATINGNHequals7'] = 0
# df_temp['d_RATINGNHequals8'] = 0
# df_temp['d_RATINGNHequals9'] = 0
# df_temp['d_RATINGNHequals10'] = 0
# df_temp.loc[(df_temp['RATINGNH'] == 1), 'd_RATINGNHequals1'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 2), 'd_RATINGNHequals2'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 3), 'd_RATINGNHequals3'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 4), 'd_RATINGNHequals4'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 5), 'd_RATINGNHequals5'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 6), 'd_RATINGNHequals6'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 7), 'd_RATINGNHequals7'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 8), 'd_RATINGNHequals8'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 9), 'd_RATINGNHequals9'] = 1
# df_temp.loc[(df_temp['RATINGNH'] == 10), 'd_RATINGNHequals10'] = 1

# print(df_temp['NHQSCRIME'].hist())
df_temp['d_NHQSCRIME'] = 0
df_temp.loc[(df_temp['NHQSCRIME'] == 1), 'd_NHQSCRIME'] = 1

# print(df_temp['NHQPCRIME'].hist())
df_temp['d_NHQPCRIME'] = 0
df_temp.loc[(df_temp['NHQPCRIME'] == 1), 'd_NHQPCRIME'] = 1

print(df_temp['UNITSIZE'].hist())
#LESS THAN 1000 SQFT
df_temp['d_UNITSIZE1'] = 0
#BETWEEN 1 and 2k SQFT
df_temp['d_UNITSIZE2'] = 0
#BETWEEN 2 and 3k SQFT
df_temp['d_UNITSIZE3'] = 0
#BETWEEN 3 and 4k SQFT
df_temp['d_UNITSIZE4'] = 0
#OVER 4k SQFT
df_temp['d_UNITSIZE5'] = 0
df_temp.loc[(df_temp['UNITSIZE'] == 1) | (df_temp['UNITSIZE'] == 2) | \
            (df_temp['UNITSIZE'] == 3), 'd_UNITSIZE1'] = 1
df_temp.loc[(df_temp['UNITSIZE'] == 4) | (df_temp['UNITSIZE'] == 5), 'd_UNITSIZE2'] = 1
df_temp.loc[(df_temp['UNITSIZE'] == 6) | (df_temp['UNITSIZE'] == 7), 'd_UNITSIZE3'] = 1
df_temp.loc[(df_temp['UNITSIZE'] == 8), 'd_UNITSIZE4'] = 1
df_temp.loc[(df_temp['UNITSIZE'] == 9), 'd_UNITSIZE5'] = 1

#NO TRASH
df_temp['d_NEARTRASH3'] = 0
#A LITTLE TRASH
df_temp['d_NEARTRASH2'] = 0
#A LOT OF TRASH
df_temp['d_NEARTRASH1'] = 0
df_temp.loc[(df_temp['NEARTRASH'] == 1) | (df_temp['NEARTRASH'] == 2), 'd_NEARTRASH1'] = 1
# df_temp.loc[(df_temp['NEARTRASH'] == 2), 'd_NEARTRASH2'] = 1
df_temp.loc[(df_temp['NEARTRASH'] == 3), 'd_NEARTRASH3'] = 1

# y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + BEDROOMS + TOTROOMS + BATHROOMS + UNITSIZE + YRBUILT + RATINGNH', data=df_temp, return_type='dataframe')
# res = sm.OLS(y, X).fit(cov_type = 'HC1')
# print(res.summary())

# B = 100
# fig, ax = plt.subplots()
# plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
# ax.set_title('MARKET VALUE')
# prob, bins, patches = ax.hist(df_temp['MARKETVAL'], bins=B, align='mid')

# y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + BEDROOMS + TOTROOMS + BATHROOMS + UNITSIZE + YRBUILT + RATINGNH', data=df_temp, return_type='dataframe')
# res = sm.OLS(y, X).fit()
# print(res.summary())

#ADJUST VARIATION IN MARKETVAL
df_temp['decimalMARKETVAL'] = df_temp['MARKETVAL']/100000

df_temp.dropna()

#SAVE TO EXCEL FOR TABLEAU VISUALIZATIONS
df_temp.to_excel("C://Users\chase\Dropbox\matthew-luey\project\stage-2\AmericanHousingSurvey\AHS_PUF_Flat\DESTRING/" + 'tableau.xlsx', index=False)

fig, ax = plt.subplots()
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
ax.set_title('MARKET VALUE')
prob, bins, patches = ax.hist(df_temp['MARKETVAL'], bins=B, align='mid')

y, X = dmatrices('decimalMARKETVAL ~ HINCP + MORTAMT + BEDROOMS + TOTROOMS + BATHROOMS + UNITSIZE + YRBUILT + RATINGNH', data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit()
print(res.summary())

#VIF
# df_temp = df_temp.dropna()

# V = df_temp[['HINCP','MORTAMT','BEDROOMS','TOTROOMS','BATHROOMS','UNITSIZE','YRBUILT','RATINGNH']]

# vif_data = pd.DataFrame()
# vif_data["Variable"] = V.columns

# vif_data["VIF"] = [variance_inflation_factor(V.values, i)
#                           for i in range(len(V.columns))]
  
# print(vif_data)

# #MULTICOLLINEARITY TESTING
# xv = df_temp['BEDROOMS'].values
# yv = df_temp['TOTROOMS'].values
# n = len(xv)

# print("Correlation coefficient = {:.3f}".format(df.BEDROOMS.corr(df.TOTROOMS)))
# #COEFFICIENT .879, DROP BEDROOMS

# xv = df_temp['TOTROOMS'].values
# yv = df_temp['UNITSIZE'].values
# n = len(xv)

# print("Correlation coefficient = {:.3f}".format(df.TOTROOMS.corr(df.UNITSIZE)))
# #COEFFICIENT .762, DROP TOTROOMS

# xv = df_temp['HINCP'].values
# yv = df_temp['MORTAMT'].values
# n = len(xv)

# print("Correlation coefficient = {:.3f}".format(df.HINCP.corr(df.MORTAMT)))
# #COEFFICIENT .224, NO ACTION TAKEN

#BEDROOMS OR TOTROOMS?
y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + BEDROOMS + d_GARAGE', data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit()
print(res.summary())

#VIF
df_temp = df_temp.dropna()

V = df_temp[['HINCP','MORTAMT','BEDROOMS','d_GARAGE']]

vif_data = pd.DataFrame()
vif_data["Variable"] = V.columns

vif_data["VIF"] = [variance_inflation_factor(V.values, i)
for i in range(len(V.columns))]

print(vif_data)

#RESIDUALS
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(res, 'HINCP', fig=fig)

fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(res, 'MORTAMT', fig=fig)

fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(res, 'BEDROOMS', fig=fig)

fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(res, 'd_GARAGE', fig=fig)

#KERNEL DENSITY
fig = plt.figure(figsize = (16, 9))

ax = sns.distplot(res.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

#NORMALITY SOLUTION?
#TAKE NATURAL LOG OF MARKETVAL
df_temp['lnMARKETVAL'] = np.log(df_temp['MARKETVAL'])

#SHOULD I INCLUDE SCHOOLS?
y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + BEDROOMS + d_NHQSCHOOL', data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit()
print(res.summary())

V = df_temp[['HINCP','MORTAMT','BEDROOMS','d_NHQSCHOOL']]

vif_data = pd.DataFrame()
vif_data["Variable"] = V.columns

vif_data["VIF"] = [variance_inflation_factor(V.values, i)
for i in range(len(V.columns))]
#THERE IS MULTICOLLINEARITY WITH BEDROOMS!!! FINISH DUMMYING UNITSIZE AND THEN USE THAT!

print(vif_data)

#SUMMARY STATS
print(df_temp['HINCP'].describe())
print(df_temp['MORTAMT'].describe())
print(df_temp['MAINTAMT'].describe())
print(df_temp['d_GARAGE'].describe())
print(df_temp['d_UNITSIZE1'].describe())
print(df_temp['d_UNITSIZE2'].describe())
print(df_temp['d_UNITSIZE3'].describe())
print(df_temp['d_UNITSIZE4'].describe())
print(df_temp['d_UNITSIZE5'].describe())
print(df_temp['d_NHQSCHOOL'].describe())
print(df_temp['d_NHQSCRIME'].describe())

#####################################################################################################
#FIRST REGRESSION
y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + MAINTAMT + d_GARAGE + d_UNITSIZE2\
                 + d_UNITSIZE3 + d_UNITSIZE4 + d_UNITSIZE5 + d_NHQSCHOOL + d_NHQSCRIME'\
                 , data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit()
print(res.summary())
#VIF

df_temp = df_temp.dropna()

V = df_temp[['HINCP','MORTAMT','MAINTAMT','d_GARAGE','d_UNITSIZE2','d_UNITSIZE3','d_UNITSIZE4','d_UNITSIZE5'\
             ,'d_NHQSCHOOL']]

vif_data = pd.DataFrame()
vif_data["Variable"] = V.columns

vif_data["VIF"] = [variance_inflation_factor(V.values, i)
for i in range(len(V.columns))]

print(vif_data)

#KDENSITY
fig = plt.figure(figsize = (16, 9))

ax = sns.distplot(res.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

#ROBUST REGRESSION
y, X = dmatrices('MARKETVAL ~ HINCP + MORTAMT + MAINTAMT + d_GARAGE + d_UNITSIZE2\
                 + d_UNITSIZE3 + d_UNITSIZE4 + d_UNITSIZE5 + d_NHQSCHOOL'\
                 , data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit(cov_type = 'HC1')
print(res.summary())

#KDENSITY
fig = plt.figure(figsize = (16, 9))

ax = sns.distplot(res.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

#################################################################################################################
#SECOND REGRESSION
y, X = dmatrices('MARKETVAL ~ MAINTAMT + d_GARAGE + d_UNITSIZE2\
                 + d_UNITSIZE3 + d_UNITSIZE4 + d_UNITSIZE5 + d_NHQSCHOOL + d_NHQSCRIME'\
                 , data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit()
print(res.summary())

#VIF
V = df_temp[['MAINTAMT','d_GARAGE','d_UNITSIZE2','d_UNITSIZE3','d_UNITSIZE4','d_UNITSIZE5'\
             ,'d_NHQSCHOOL','d_NHQSCRIME']]

vif_data = pd.DataFrame()
vif_data["Variable"] = V.columns

vif_data["VIF"] = [variance_inflation_factor(V.values, i)
for i in range(len(V.columns))]

print(vif_data)

#KDENSITY
fig = plt.figure(figsize = (16, 9))

ax = sns.distplot(res.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

# #NORMALITY SOLUTION ATTEMPT
# y, X = dmatrices('lnMARKETVAL ~ MAINTAMT + d_GARAGE + d_UNITSIZE2\
#                  + d_UNITSIZE3 + d_UNITSIZE4 + d_UNITSIZE5 + d_NHQSCHOOL + d_NHQSCRIME',\
#                      data=df_temp, return_type='dataframe')
# lnres = sm.OLS(y, X).fit()
# print(lnres.summary())

# fig = plt.figure(figsize = (16, 9))

# ax = sns.distplot(lnres.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

# ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
# ax.set_xlabel("Residuals")

#ROBUST REGRESSION
y, X = dmatrices('MARKETVAL ~ MAINTAMT + d_GARAGE + d_UNITSIZE2\
                 + d_UNITSIZE3 + d_UNITSIZE4 + d_UNITSIZE5 + d_NHQSCHOOL + d_NHQSCRIME'\
                 , data=df_temp, return_type='dataframe')
res = sm.OLS(y, X).fit(cov_type = 'HC1')
print(res.summary())