#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import matplotlib.pyplot as plt
import pickle

import bamboolib

from tqdm import tqdm

import plotly
import plotly.offline
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import HTML
from IPython.core.display import display, HTML
import copy

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.linear_model import ElasticNet, Lasso, LassoLarsIC, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit,train_test_split, cross_validate, cross_val_score, validation_curve

import plotly.express as px
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as stats

import pingouin as pg

import logging
import logging.config

from utils.misc import LogWrapper, BlockingTimeSeriesSplit, SMWrapper, StandardScalerClipper

import chart_studio.plotly as py
import plotly.graph_objs as go

from utils.plotlyhelper import plotly_fig2json


# In[2]:


logging.config.fileConfig('../../logs/setting.log', disable_existing_loggers=False)
_logger = logging.getLogger(__name__)
logger = LogWrapper(_logger)

def log(*args, logtype='debug', sep=' '):
    getattr(logger, logtype)(sep.join(str(a) for a in args))

# Plotly settings
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[3]:


init_notebook_mode()


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # 1. Load datasets

# In[19]:


bf_filename = '../../data/processed/base_assets_M.pkl'
bf_w_filename = '../../data/processed/base_assets_W.pkl'
mf_filename = '../../data/processed/macro_factors_M.pkl'
mf_w_filename = '../../data/processed/macro_factors_W.pkl'
# phase_filename = '../../data/processed/phase_M.pkl'
# phase_w_filename = '../../data/processed/phase_W.pkl'


# In[20]:


_freq = 'M'


# #### Set a start date and end date

# In[22]:


ds_bf = pd.read_pickle(bf_filename) if _freq == 'M' else pd.read_pickle(bf_w_filename)
# ds_mf = pd.read_pickle(mf_filename) if _freq == 'M' else pd.read_pickle(mf_w_filename)
# phase = pd.read_pickle(phase_filename) if _freq == 'M' else pd.read_pickle(phase_w_filename)
# phase.name = 'phase'



# In[14]:


inputs = pd.read_csv('../../data/raw/inputs.log', index_col='Date', parse_dates=True)
inputs.index = inputs.index.to_period(_freq)


# In[29]:

phase = inputs.Phase
start_dt = min(phase.index)
end_dt = max(phase.index)



# In[32]:


ds_mf = inputs.iloc[:, 1:]



# # 2. Preprocessing

# #### Calculate `m`-length rolling returns.

# In[8]:


bf = pd.DataFrame(columns=ds_bf.columns)
lb_range = range(1, 25) if _freq == 'M' else [1] + list(range(4, 108, 4))

for m in tqdm(lb_range):
    rolling_bf = ds_bf.rolling(window=m).sum()
    rolling_bf['lookback'] = int(m)
    bf = pd.concat((bf, rolling_bf), axis=0)[rolling_bf.columns]

# Add a phase column.
bf = pd.merge(bf, phase, how='right', left_index=True, right_index=True)


# #### From EDAV, we know that `best_lookback` is 12 weeks for the weekly model.

# In[9]:


best_lookback = 3 if _freq == 'M' else 12


# #### Split the dataset
# - `8:2` split: fixed past data

# In[10]:


test_size = 0.2


# In[11]:


X = pd.merge(ds_mf.loc[start_dt:end_dt].copy(), phase, how='inner', left_index=True, right_index=True)
y_train_dic = {}
y_test_dic = {}

for b in tqdm(ds_bf.columns):
    y_train_dic[b] = {}
    y_test_dic[b] = {}
    
    for m in lb_range:
        y = bf.loc[bf.lookback == m, b].copy()
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            shuffle=False,
                                                            test_size=test_size)
        y_train_dic[b][m] = y_train
        y_test_dic[b][m] = y_test


# - `Walk forward split`
#   - Sliding window size `sliding_window`: 20 years

# In[12]:


sliding_window = 12*10 if _freq == 'M' else 52*10


# In[13]:


X_train_wf_dic = {}
X_test_wf_dic = {}
y_train_wf_dic = {}
y_test_wf_dic = {}

# for b in tqdm(ds_bf.columns):
#     y_train_wf_dic[b] = {}
#     y_test_wf_dic[b] = {}
for t, idx in tqdm(enumerate(range(sliding_window, X.shape[0]-1))):
    y = bf.loc[bf.lookback == best_lookback, :].drop(['lookback'], axis=1).copy()
    X_train_wf_dic[t], X_test_wf_dic[t], y_train_wf_dic[t], y_test_wf_dic[t] = X.iloc[t:t+sliding_window], X.iloc[t+sliding_window:t+sliding_window+1], y.iloc[t:t+sliding_window], y.iloc[t+sliding_window:t+sliding_window+1]


# In[14]:


y_test_wf_dic[0]


# # 3. Model training
# # ## 3.1) Static inputs: fixed past data from 1955 to 2007.
# # ## `ML-basic` model.
# # - Step 1: Apply a variant of Two-Stage Least-Squares Instrumental Variables estimation approach.
# #   - Stage 1: Supervised PCAing base assets to get fitted macro factors so that we can reduce measurement errors within the macroeconomic variables.
# #   - Stage 2: Run OLS post-t Lasso on the fitted macro factors over each base asset as *y* to get macro-factor loadings.
# # - Step 2: Run multivariate OLS where $X$ is macro factors and $y$ is a base asset by replacing the OLS betas with the factor loadings $\mathbf{B}$. Then, we have FMP weights $\mathbf{W_K}$.
# #   - $\mathbf{W_K=B(B^TB)^{-1}}$

# # ## Step 1: Get macro-factor loadings $\mathbf{B}$

# # ### We do supervised PCA and then OLS post-t Lasso.

# # #### Integrate `phase` into a dataset through one-hot encoding

# # In[15]:


categorical_columns = ['phase']
onehot_encoding = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), categorical_columns),
    remainder='passthrough')


# # In[16]:


# bf_train_trial = pd.DataFrame().from_dict({b:y_train_dic[b][best_lookback] for b in ds_bf.columns})

# # Get principal components of baset assets
# # Merge 'phase' with the base asset frame for classification to be integrated in this model.
# bf_train_trial = bf_train_trial.merge(phase, how='left', left_index=True, right_index=True)


# # Apply PCA to base asset returns
# pca_pipe = Pipeline([('onehot_encoding', onehot_encoding), ('pca', PCA())])
# bf_train = bf_train_trial.copy()
# y_pca = pca_pipe.fit_transform(bf_train)


# # Apply *Lasso* to `y_pca`
# param_grid = {'estimator__regressor__criterion': ['aic', 'bic']}

# model_lasso = {}
# fitted_mf_train = {}  # Fitted macro factors (in the training set)

# for f in ds_mf.columns:
#     # This selection comes from GridSearchCV at an EDAV stage.
#     opt_criterion = 'aic' if f == 'GRTH' else 'bic'
    
#     scaler_x = StandardScaler()
# #     scaler_y = StandardScaler()
#     scaler_y = StandardScalerClipper(-3, 3)

#     # We project each macro factor on a space spanned by principal components of base assets.
#     # -> X_train, named as `X_pc_train`, should be `y_pca[m]`.
#     X_pc_train = y_pca

#     # We want to find a fitted macro factor.
#     # -> y_train should be X_train[GRTH|INFL|UNCR]
#     y_train = X_train[f]

#     model_lasso[f] = Pipeline([
#         ('standardizing_X', scaler_x),
#         ('estimator',
#          TransformedTargetRegressor(regressor=LassoLarsIC(criterion=opt_criterion),
#                                     transformer=scaler_y))
#     ])


#     model_lasso[f].fit(X_pc_train, y_train)
#     fitted_mf_train[f] = model_lasso[f].predict(X_pc_train)

# # Save the fitted macro factors as DataFrame.
# fitted_mf_train = pd.DataFrame().from_dict(fitted_mf_train)

# # Get fitted macro factors using trained models.
# bf_test_trial = pd.DataFrame().from_dict({b:y_test_dic[b][best_lookback] for b in ds_bf.columns})
# bf_test_trial = bf_test_trial.merge(phase, how='left', left_index=True, right_index=True)

# # We use the same pca instances trained on the training set to prevent any look-ahead bias.
# y_pca_test = pca_pipe.transform(bf_test_trial)


# # Predict fitted macro factors in the test set.
# fitted_mf_test = {}
# for f in ds_mf.columns:
#     # As we did in the training process above, we take principal components extracted from trained-PCA instances, y_pca_test[m], as our spanned-space, X_pc_test.
#     X_pc_test = y_pca_test

#     # `fitted_mf_test[f][m]` contains fitted factor returns of 'f' factor for an 'm' lookback period.
#     fitted_mf_test[f] = model_lasso[f].predict(X_pc_test)

# fitted_mf_test = pd.DataFrame().from_dict(fitted_mf_test)


# X_fit_train = fitted_mf_train.to_numpy()
# X_fit_test = fitted_mf_test.to_numpy()


# # Stage 2: Apply OLS Post-t Lasso to each of the selected model, `model_lasso[f][m]`
# ## the threshold to drop a coefficient set to be 0.05.
# model_sqrt_lasso = {}
# coef_sqrt_lasso = {}
# y_pred = {}

# for b in ds_bf.columns:
#     scaler_x = StandardScalerClipper(-3,3)
#     scaler_y = StandardScaler()

#     y_train = bf_train_trial[b].to_numpy()
#     y_test = bf_test_trial[b].to_numpy()

#     model_sqrt_lasso[b] = Pipeline([
#         ('standardizing_X', scaler_x),
#         ('estimator', TransformedTargetRegressor(regressor=SMWrapper(model_class=sm.OLS, lasso_t=0.05),
#                                     transformer=scaler_y))
#     ])

#     model_sqrt_lasso[b].fit(X_fit_train, y_train)
#     coef_sqrt_lasso[b] = model_sqrt_lasso[b].named_steps['estimator'].regressor_.results_.params[1:]
#     y_pred[b] = model_sqrt_lasso[b].predict(X_fit_test)


# # In[17]:


# B_df = pd.DataFrame().from_dict(coef_sqrt_lasso, orient='index', columns=ds_mf.columns)


# # #### Now we have $\mathbf{B}$ as follows.

# # In[18]:


# B_df.style.format('{:.2f}')


# # ## Step 2: Compute an FMP weight vector $\mathbf{W_K}$

# # #### Finally we can calculate an FMP weight vector $\mathbf{W_K}$:
# # - $\mathbf{W_K} = \mathbf{\Omega^{-1}B(B^T \Omega^{-1}B)^{-1}}$ and this can be further simplified depending on a choice of covariance matrix of base assets $\mathbf{\Omega}$:
# #   - 1) $\mathbf{\Omega}=\sigma\mathbf{I_N}$: base assets are `uncorrelated` with `constant variance` over time.
# #   - 2) $\mathbf{\Omega}=Diag(\sigma^2)$: base assets are `uncorrelated`.
# #   - 3) `Unconstrained` $\mathbf{\Omega}$.

# # In[19]:


# B = B_df.to_numpy()
# W = B@np.linalg.inv(B.T@B)
# W_df = pd.DataFrame(W, index=ds_bf.columns, columns=ds_mf.columns)


# # In[20]:


# W_df.style.format('{:.2f}')


# # In[21]:


# np.sum(np.abs(W_df))


# # In[22]:


# W_df.style.format('{:.2f}')


# # In[23]:


# np.sum(np.abs(W_df))


# # #### Calculate a macro factor return matrix `mf_rt`.

# # In[24]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# import quantstats as qs


# # In[25]:


# start_test_dt = bf_test_trial.index[0]
# bf_test = ds_bf[start_test_dt:]
# fmp_rt = bf_test@W_df
# fmp_rt = fmp_rt.set_index(fmp_rt.index.to_timestamp(how='E'))


# # In[26]:


# scaler = StandardScaler()
# mf_rt = X_test.loc[start_test_dt:, ds_mf.columns].copy()
# mf_rt = pd.DataFrame(mf_rt, index=fmp_rt.index, columns=ds_mf.columns)


# # In[27]:


# lb_range = [1] + list(range(4,56,4))
# fmp_cum = {cum:fmp_rt.rolling(window=cum).sum() for cum in lb_range}
# window_sz = [52, 104]


# # In[28]:


# r_corr = {}
# cross_corr_fixed = None
# for cum in tqdm(fmp_cum.keys()):
#     r_corr[cum] = {}
#     for w in window_sz:
#         r_corr[cum][w] = {}
#         for r_idx in range(len(mf_rt.index)-w):
#             c = pd.merge(fmp_cum[cum][r_idx:r_idx+w], mf_rt[r_idx:r_idx+w], how='inner', left_index=True, right_index=True, suffixes=('_fmp', '_mf')).corr()
#             r_corr[cum][w][mf_rt.index[r_idx+w]] = c.loc['GRTH_fmp', 'GRTH_mf'], c.loc['INFL_fmp', 'INFL_mf'], c.loc['UNCR_fmp', 'UNCR_mf']
        
#         corr_fmp_mf = pd.DataFrame().from_dict(r_corr[cum][w], orient='index', columns=ds_mf.columns)
#         if cross_corr_fixed is None:
#             cross_corr_fixed = corr_fmp_mf.copy()
#             cross_corr_fixed['lookback'] = cum
#             cross_corr_fixed['window'] = w
#         else:
#             corr_fmp_mf['lookback'] = cum
#             corr_fmp_mf['window'] = w
#             cross_corr_fixed = pd.concat((cross_corr_fixed, corr_fmp_mf.copy()))


# # In[29]:


# cross_corr_fixed.index.name='date'
# cross_corr_fixed = cross_corr_fixed.reset_index()
# cross_corr_fixed['date'] = cross_corr_fixed['date'].dt.strftime('%Y-%m-%d')


# # ### Save works: `Fixed07` model

# # - `corr_filename`: Cross correlations between `n`-week cumulative FMP returns and macro factors.
# # - `fmp_rt_filename`: FMP returns.
# # - `mf_rt_filename`: Macro factor (returns).
# # - `mf_wt_filename`: Macro factor weights. **FILAL RESULTS of this model.**

# # In[30]:


# corr_filename = '../../reports/tables/fixed_corr'
# fmp_rt_filename = '../../reports/tables/fixed_fmp_rt'
# mf_rt_filename = '../../reports/tables/fixed_mf_rt'
# fmp_wt_filename = '../../reports/tables/fixed_fmp_wt'


# # In[31]:


# cross_corr_fixed.to_pickle(corr_filename + '.pkl')
# fmp_rt.to_pickle(fmp_rt_filename + '.pkl')
# mf_rt.to_pickle(mf_rt_filename + '.pkl')
# W_df.to_pickle(fmp_wt_filename + '.pkl')


# ## 3.2) Sliding-window inputs

# In[32]:


input_sz = len(y_train_wf_dic.keys())


# In[33]:


B_df = {}

for t in tqdm(range(input_sz)):
    bf_train_trial = y_train_wf_dic[t]
    
    # Apply PCA to base asset returns
    pca_pipe = Pipeline([('onehot_encoding', onehot_encoding), ('pca', PCA())])
    bf_train = bf_train_trial.copy()
    y_pca = pca_pipe.fit_transform(bf_train)

    model_lasso = {}
    fitted_mf_train = {}  # Fitted macro factors (in the training set)

    for f in ds_mf.columns:
        # This selection comes from GridSearchCV at an EDAV stage.
        opt_criterion = 'aic' if f == 'GRTH' else 'bic'

        scaler_x = StandardScaler()
        scaler_y = StandardScalerClipper(-3, 3)

        # We project each macro factor on a space spanned by principal components of base assets.
        # -> X_train, named as `X_pc_train`, should be `y_pca[m]`.
        X_pc_train = y_pca

        # We want to find a fitted macro factor.
        # -> y_train should be X_train[GRTH|INFL|UNCR]
        y_train = X_train_wf_dic[t][f]

        model_lasso[f] = Pipeline([
            ('standardizing_X', scaler_x),
            ('estimator',
             TransformedTargetRegressor(regressor=LassoLarsIC(criterion=opt_criterion),
                                        transformer=scaler_y))
        ])

        model_lasso[f].fit(X_pc_train, y_train)
        fitted_mf_train[f] = model_lasso[f].predict(X_pc_train)

    # Save the fitted macro factors as DataFrame.
    fitted_mf_train = pd.DataFrame().from_dict(fitted_mf_train)

    # Get fitted macro factors using trained models.
    bf_test_trial = pd.DataFrame(y_test_wf_dic[t])

    # We use the same pca instances trained on the training set to prevent any look-ahead bias.
    y_pca_test = pca_pipe.transform(bf_test_trial)

    # Predict fitted macro factors in the test set.
    fitted_mf_test = {}
    for f in ds_mf.columns:
        # As we did in the training process above, we take principal components extracted from trained-PCA instances, y_pca_test[m], as our spanned-space, X_pc_test.
        X_pc_test = y_pca_test

        # `fitted_mf_test[f][m]` contains fitted factor returns of 'f' factor
        fitted_mf_test[f] = model_lasso[f].predict(X_pc_test)

    fitted_mf_test = pd.DataFrame().from_dict(fitted_mf_test)

    X_fit_train = fitted_mf_train.to_numpy()
    X_fit_test = fitted_mf_test.to_numpy()
    
    # Stage 2: Apply OLS Post-t Lasso to each of the selected model, `model_lasso[f][m]`
    ## the threshold to drop a coefficient set to be 0.05.
    model_sqrt_lasso = {}
    coef_sqrt_lasso = {}
    
    for b in ds_bf.columns:
        scaler_x = StandardScalerClipper(-3, 3)
        scaler_y = StandardScaler()

        y_train = bf_train_trial[b].to_numpy(dtype=np.float)

        model_sqrt_lasso[b] = Pipeline([
            ('standardizing_X', scaler_x),
            ('estimator', TransformedTargetRegressor(regressor=SMWrapper(model_class=sm.OLS, lasso_t=0.05),
                                        transformer=scaler_y))
        ])

        try:
            model_sqrt_lasso[b].fit(X_fit_train, y_train)
            coef_sqrt_lasso[b] = model_sqrt_lasso[b].named_steps['estimator'].regressor_.results_.params[1:]
        except Exception as e:
            print('You might want to increase the size of your test set.\n')
            print('An exception occurs:', e)
            print(t)
            print(b)
            print(X_fit_train)
            print(y_train)
    
    # X_test_wf_dic[t].index[0] is an index of the following week.
    # e.g.: We train on a training set of pre-June 2020 data and
    # save the result in B_df[1st week of July 2020].
    B_df[X_test_wf_dic[t].index[0]] = pd.DataFrame().from_dict(coef_sqrt_lasso, orient='index', columns=ds_mf.columns)


# #### Calculate weight vector $\mathbf{W}$ in `W_df`  and FMP returns in `fmp_rt`

# In[34]:


W_df = {}
fmp_rt = {}
for k in tqdm(B_df.keys()):
    B = B_df[k].to_numpy()
    W = B@np.linalg.inv(B.T@B)
    W_df[k] = pd.DataFrame(W, index=ds_bf.columns, columns=ds_mf.columns)
    fmp_rt[k] = ds_bf.loc[k]@W_df[k]


# In[35]:


fmp_rt = pd.DataFrame().from_dict(fmp_rt, orient='index')
scaler = StandardScaler()
mf_rt = X.loc[fmp_rt.index[0]:fmp_rt.index[-1], ds_mf.columns].copy()
fmp_rt.set_index(fmp_rt.index.to_timestamp(how='E'), inplace=True)
mf_rt.set_index(mf_rt.index.to_timestamp(how='E'), inplace=True)
lb_range = [1] + list(range(4,56,4))
fmp_cum = {cum:fmp_rt.rolling(window=cum).sum() for cum in lb_range}
window_sz = [52, 104]


# ### For comparison, let's produce FMP weights on a new test set where the latest phase is changed to`Recession` from `Robust Growth`

# In[101]:


latest_X_train = X_train_wf_dic[max(X_train_wf_dic.keys())].copy()
latest_X_train.at[latest_X_train.index[-1], 'phase'] = 'Recession'

latest_y_train = y_train_wf_dic[max(y_train_wf_dic.keys())].copy()
latest_y_train.at[latest_y_train.index[-1], 'phase'] = 'Recession'


# In[98]:


latest_X_test = X_test_wf_dic[max(X_train_wf_dic.keys())].copy()
latest_X_test.at[latest_X_test.index[-1], 'phase'] = 'Recession'

latest_y_test = y_test_wf_dic[max(y_test_wf_dic.keys())].copy()
latest_y_test.at[latest_y_test.index[-1], 'phase'] = 'Recession'


# In[134]:


B_df_latest = {}

# Apply PCA to base asset returns
bf_train_latest = latest_y_train.copy()
y_pca = pca_pipe.transform(bf_train_latest)

fitted_mf_train_latest = {}  # Fitted macro factors (in the training set)

for f in ds_mf.columns:
    # This selection comes from GridSearchCV at an EDAV stage.
    opt_criterion = 'aic' if f == 'GRTH' else 'bic'


    # We project each macro factor on a space spanned by principal components of base assets.
    # -> X_train, named as `X_pc_train`, should be `y_pca[m]`.
    X_pc_train = y_pca

    # We want to find a fitted macro factor.
    # -> y_train should be X_train[GRTH|INFL|UNCR]
    y_train = latest_X_train[f]

    fitted_mf_train_latest[f] = model_lasso[f].predict(X_pc_train)

# Save the fitted macro factors as DataFrame.
fitted_mf_train_latest = pd.DataFrame().from_dict(fitted_mf_train_latest)

# Get fitted macro factors using trained models.
# bf_test_trial = latest_y_test.copy()

# We use the same pca instances trained on the training set to prevent any look-ahead bias.
# y_pca_test = pca_pipe.transform(bf_test_trial)

# # Predict fitted macro factors in the test set.
# fitted_mf_test_latest = {}
# for f in ds_mf.columns:
#     # As we did in the training process above, we take principal components extracted from trained-PCA instances, y_pca_test[m], as our spanned-space, X_pc_test.
#     X_pc_test = y_pca_test

#     # `fitted_mf_test[f][m]` contains fitted factor returns of 'f' factor
#     fitted_mf_test_latest[f] = model_lasso[f].predict(X_pc_test)

# fitted_mf_test_latest = pd.DataFrame().from_dict(fitted_mf_test_latest)

X_fit_train = fitted_mf_train_latest.to_numpy()
# X_fit_test = fitted_mf_test_latest.to_numpy()

# Stage 2: Apply OLS Post-t Lasso to each of the selected model, `model_lasso[f][m]`
## the threshold to drop a coefficient set to be 0.05.
coef_latest = {}

for b in ds_bf.columns:
    y_train = bf_train_latest[b].to_numpy(dtype=np.float)

    try:
        model_sqrt_lasso[b].fit(X_fit_train, y_train)
        coef_latest[b] = model_sqrt_lasso[b].named_steps['estimator'].regressor_.results_.params[1:]
    except Exception as e:
        print('You might want to increase the size of your test set.\n')
        print('An exception occurs:', e)
        print(t)
        print(b)
        print(X_fit_train)
        print(y_train)

B_df_latest[latest_X_test.index[0]] = pd.DataFrame().from_dict(coef_latest, orient='index', columns=ds_mf.columns)


# In[138]:


B_df_latest[max(B_df_latest.keys())]


# In[137]:


B_df[max(B_df.keys())]


# #### Calculate new FMP weights with `B_df_latest`

# In[139]:


W_df_latest = {}
for k in tqdm(B_df_latest.keys()):
    B = B_df_latest[k].to_numpy()
    W = B@np.linalg.inv(B.T@B)
    W_df_latest[k] = pd.DataFrame(W, index=ds_bf.columns, columns=ds_mf.columns)


# In[142]:


W_df_latest[k].to_pickle('../../reports/tables/fmp_wt_spot_if_recession.pkl')


# #### Compute correlation matrice between various-length FMP returns `cum` and factor returns.
# - Lookback-window is `window_sz` 

# In[40]:


r_corr = {}
cross_corr_rolling = None
for cum in tqdm(fmp_cum.keys()):
    r_corr[cum] = {}
    for w in window_sz:
        r_corr[cum][w] = {}
        for r_idx in range(len(mf_rt.index)-w):
            c = pd.merge(fmp_cum[cum][r_idx:r_idx+w], mf_rt[r_idx:r_idx+w], how='inner', left_index=True, right_index=True, suffixes=('_fmp', '_mf')).corr()
            r_corr[cum][w][mf_rt.index[r_idx+w]] = c.loc['GRTH_fmp', 'GRTH_mf'], c.loc['INFL_fmp', 'INFL_mf'], c.loc['UNCR_fmp', 'UNCR_mf']
        
        corr_fmp_mf = pd.DataFrame().from_dict(r_corr[cum][w], orient='index', columns=ds_mf.columns)
        if cross_corr_rolling is None:
            cross_corr_rolling = corr_fmp_mf.copy()
            cross_corr_rolling['lookback'] = cum
            cross_corr_rolling['window'] = w
        else:
            corr_fmp_mf['lookback'] = cum
            corr_fmp_mf['window'] = w
            cross_corr_rolling = pd.concat((cross_corr_rolling, corr_fmp_mf.copy()))


# In[41]:


cross_corr_rolling.index.name='date'
cross_corr_rolling = cross_corr_rolling.reset_index()
cross_corr_rolling['date'] = cross_corr_rolling['date'].dt.strftime('%Y-%m-%d')


# In[42]:


def show_corr(n_week):
    fig = plotly.tools.make_subplots(rows=3, cols=3, horizontal_spacing = 0.05, vertical_spacing = 0.1,
                                     subplot_titles=('Fixed 07: Growth', 'Inflation', 'Uncertainty', 'Rolling20: Growth', 'Inflation', 'Uncertainty',
                                                    'LT-Rolling20: Growth', 'Inflation', 'Uncertainty'))

    # Fixed 2007. Short-term plots.
    fig.add_trace(
        go.Heatmap(
            name='Correlation (Fixed07)',
            z=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'GRTH'],
            x=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'date'],
            y=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 1, 1)

    fig.add_trace(
        go.Heatmap(
            name='Correlation (Fixed07)',
            z=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'INFL'],
            x=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'date'],
            y=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 1, 2)

    fig.add_trace(
        go.Heatmap(
            name='Correlation (Fixed07)',
            z=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'UNCR'],
            x=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'date'],
            y=cross_corr_fixed.loc[cross_corr_fixed.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 1, 3)
    
    # Rolling 20 years. Short-term plots.
    fig.add_trace(
        go.Heatmap(
            name='Correlation (R20)',
            z=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'GRTH'],
            x=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'date'],
            y=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 2, 1)


    fig.add_trace(
        go.Heatmap(
            name='Correlation (R20)',
            z=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'INFL'],
            x=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'date'],
            y=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 2, 2)

    fig.add_trace(
        go.Heatmap(
            name='Correlation (R20)',
            z=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'UNCR'],
            x=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'date'],
            y=cross_corr_rolling.loc[np.logical_and(cross_corr_rolling.date>=start_test_dt.to_timestamp(how='E').strftime('%Y-%m-%d'), cross_corr_rolling.window==n_week), 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 2, 3)
    
    
    # Rolling 20 years. Long-term plots.
    fig.add_trace(
        go.Heatmap(
            name='Correlation (LT-R20)',
            z=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'GRTH'],
            x=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'date'],
            y=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 3, 1)

    fig.add_trace(
        go.Heatmap(
            name='Correlation (LT-R20)',
            z=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'INFL'],
            x=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'date'],
            y=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 3, 2)

    fig.add_trace(
        go.Heatmap(
            name='Correlation (LT-R20)',
            z=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'UNCR'],
            x=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'date'],
            y=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'lookback'],
            colorscale='RdBu', zmin=-1, zmax=1
        ), 3, 3)


    fig['layout'].update(height=800, width=1500, title=str(n_week) + '-week Correlations (FMP, Macro factors)', template='plotly_white',
                         yaxis=dict(title='n-week FMP returns')
                        )
    return fig


# In[43]:


fig = show_corr(52)
fig.show()


# In[44]:


fig_name = '../../reports/figures/r20_corr_w52_clip_plotly.json'
plotly_fig2json(fig, fig_name)


# In[45]:


fig = show_corr(104)
fig.show()


# In[46]:


fig_name = '../../reports/figures/r20_corr_w104_clip_plotly.json'
plotly_fig2json(fig, fig_name)


# ### Save works: `R20` model

# - `corr_filename`: Cross correlations between `n`-week cumulative FMP returns and macro factors.
# - `fmp_rt_filename`: FMP returns.
# - `mf_rt_filename`: Macro factor (returns).
# - `mf_wt_filename`: Macro factor weights. **FILAL RESULTS of this model.**

# In[47]:


corr_filename = '../../reports/tables/r20_corr_clip'
fmp_rt_filename = '../../reports/tables/r20_fmp_rt_clip'
mf_rt_filename = '../../reports/tables/r20_mf_rt_clip'
mf_wt_filename = '../../reports/tables/r20_fmp_wt_clip'


# In[48]:


cross_corr_rolling.to_pickle(corr_filename + '.pkl')
fmp_rt.to_pickle(fmp_rt_filename + '.pkl')
mf_rt.to_pickle(mf_rt_filename + '.pkl')

with open(mf_wt_filename + '.pkl', 'wb') as f:
    pickle.dump(W_df, f)


# In[ ]:




