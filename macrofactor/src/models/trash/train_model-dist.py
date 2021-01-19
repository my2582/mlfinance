#!/usr/bin/env python
# coding: utf-8

# In[234]:


import numpy as np
import pandas as pd
import os
from pandas.tseries.offsets import MonthEnd
from datetime import date
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import pickle

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

import logging
import logging.config

from utils.misc import LogWrapper, BlockingTimeSeriesSplit, SMWrapper, StandardScalerClipper, get_start_end_dates, get_nonexistant_path

import chart_studio.plotly as py
import plotly.graph_objs as go

from utils.plotlyhelper import plotly_fig2json, plotly_multi_shades


# In[2]:


# Plotly settings
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[3]:


init_notebook_mode()


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # 1. Load datasets

# In[5]:


bf_filename = '../../data/processed/base_assets_M.pkl'
bf_w_filename = '../../data/processed/base_assets_W.pkl'


# In[6]:


_freq = 'W'


# #### Set a start date and end date

# In[7]:


ds_bf = pd.read_pickle(bf_filename) if _freq == 'M' else pd.read_pickle(bf_w_filename)


# In[8]:


inputs = pd.read_csv('../../data/raw/inputs.log', index_col='Date', parse_dates=True)
inputs.index = inputs.index.to_period('M') if _freq == 'M' else inputs.index.to_period('W-FRI')
phase = inputs.Phase
phase.name = 'phase'
start_dt = min(phase.index)
end_dt = max(phase.index)


# In[9]:


phase_filename = '../../data/processed/phase_' + _freq
inputs.Phase.to_pickle(phase_filename + '.pkl')


# In[10]:


ds_mf = inputs.loc[:, inputs.columns != 'Phase']


# In[11]:


ds_mf


# In[12]:


phase.value_counts()


# In[13]:


ds_bf


# # 2. Preprocessing

# ### 2-1) Settings
# #### From EDAV, we know that `best_lookback` is 12 weeks for the weekly model.

# In[14]:


best_lookback = 3 if _freq == 'M' else 12


# #### Split the dataset
# - `8:2` split: fixed past data

# In[15]:


test_size = 0.3


# ### Some calculations

# #### Calculate `m`-length rolling returns.

# In[16]:


bf = pd.DataFrame(columns=ds_bf.columns)
lb_range = range(1, 25) if _freq == 'M' else [1] + list(range(4, 108, 4))

for m in tqdm(lb_range):
    rolling_bf = ds_bf.rolling(window=m).sum()
    rolling_bf['lookback'] = int(m)
    bf = pd.concat((bf, rolling_bf), axis=0)[rolling_bf.columns]

# Add a phase column.
bf = pd.merge(bf, phase, how='right', left_index=True, right_index=True)


# In[17]:


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
#   - Sliding window size `sliding_window`: 10 years

# In[18]:


sliding_window = 12*10 if _freq == 'M' else 52*10


# In[19]:


X_train_wf_dic = {}
X_test_wf_dic = {}
y_train_wf_dic = {}
y_test_wf_dic = {}

# for b in tqdm(ds_bf.columns):
#     y_train_wf_dic[b] = {}
#     y_test_wf_dic[b] = {}
for t, idx in tqdm(enumerate(range(sliding_window, X.shape[0]))):
    y = bf.loc[bf.lookback == best_lookback, :].drop(['lookback'], axis=1).copy()
    X_train_wf_dic[t], X_test_wf_dic[t], y_train_wf_dic[t], y_test_wf_dic[t] = X.iloc[t:t+sliding_window], X.iloc[t+sliding_window:t+sliding_window+1], y.iloc[t:t+sliding_window], y.iloc[t+sliding_window:t+sliding_window+1]


# In[20]:


y_test_wf_dic[0]


# # 3. Model training

# ## `ML-basic` model.
# - Step 1: Apply a variant of Two-Stage Least-Squares Instrumental Variables estimation approach.
#   - Stage 1: Supervised PCAing base assets to get fitted macro factors so that we can reduce measurement errors within the macroeconomic variables.
#   - Stage 2: Run OLS post-t Lasso on the fitted macro factors over each base asset as *y* to get macro-factor loadings $\mathbf{B}$.
# - Step 2: Run multivariate OLS where $X$ is macro factors and $y$ is a base asset by replacing the OLS betas with the factor loadings $\mathbf{B}$. Then, we have FMP weights $\mathbf{W_K}$.
#   - $\mathbf{W_K=\Omega^{-1}B(B^T\Omega^{-1}B)^{-1}B_K=B(B^TB)^{-1}}$
#     - $\mathbf{\Omega}=\sigma\mathbf{I_N}$ (Uncorrrelated assets with constant variance)
#     - $\mathbf{B_K=I_K}$ (identity matrix)

# #### Integrate `phase` into a dataset through one-hot encoding

# In[21]:


categorical_columns = ['phase']
onehot_encoding = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), categorical_columns),
    remainder='passthrough')


# ## Step 1: Get macro-factor loadings $\mathbf{B}$

# ### We do supervised PCA and then OLS post-t Lasso.

# In[22]:


input_sz = len(y_train_wf_dic.keys())


# In[23]:


X_train_wf_dic[t]


# In[24]:


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
        # This selection comes from GridSearchCV at an EDAV stage. No more used because macro factors have changed since Oct 2020.
        opt_criterion = 'aic' if f == 'Growth' else 'bic'

        scaler_x = StandardScaler()
        scaler_y = StandardScalerClipper(zmin=-3, zmax=3)

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
                                        transformer=scaler_y, check_inverse=False))  # scaler_y clips z-values outside [-3, 3] to either -3 or 3, so no inversable.
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
        scaler_x = StandardScalerClipper(zmin=-3, zmax=3)
        scaler_y = StandardScaler()

        y_train = bf_train_trial[b].to_numpy(dtype=np.float)

        model_sqrt_lasso[b] = Pipeline([
            ('standardizing_X', scaler_x),
            ('estimator', TransformedTargetRegressor(regressor=SMWrapper(model_class=sm.OLS, lasso_t=0.05, fit_intercept=True, refit=True),
                                        transformer=scaler_y))
        ])

        try:
            model_sqrt_lasso[b].fit(X_fit_train, y_train)
            coef_sqrt_lasso[b] = model_sqrt_lasso[b].named_steps['estimator'].regressor_.results_.params[1:]  # We don't use an intercept, which is in 0th element.
            num_params = coef_sqrt_lasso['DMEQ'].shape[0]
            if num_params == 6:
                print('Only six params found. We need seven. X_fit_train is:', X_fit_train)
        except Exception as e:
            print('You might want to increase the size of your test set.\n')
            print('An exception occurs:', e)
    
    # X_test_wf_dic[t].index[0] is an index of the following week.
    # e.g.: We train on a training set of pre-June 2020 data and
    # save the result in B_df[1st week of July 2020].
    B_df[X_test_wf_dic[t].index[0]] = pd.DataFrame().from_dict(coef_sqrt_lasso, orient='index', columns=ds_mf.columns)


# #### Now we have $\mathbf{B}$ as follows.

# - New inputs on Nov 26, 2020.

# In[25]:


B_df[max(B_df.keys())].style.format('{:.2f}')


# In[26]:


B_df[max(B_df.keys())].style.format('{:.2f}')


# In[27]:


print('Step 1: macro factor loadings B are calculated.')


# ## Step 2: Compute an FMP weight vector $\mathbf{W_K}$

# #### Finally we can calculate an FMP weight vector $\mathbf{W_K}$:
# - $\mathbf{W_K} = \mathbf{\Omega^{-1}B(B^T \Omega^{-1}B)^{-1}}$ and this can be further simplified depending on a choice of covariance matrix of base assets $\mathbf{\Omega}$:
#   - 1) $\mathbf{\Omega}=\sigma\mathbf{I_N}$: base assets are `uncorrelated` with `constant variance` over time.
#   - 2) $\mathbf{\Omega}=Diag(\sigma^2)$: base assets are `uncorrelated`.
#   - 3) `Unconstrained` $\mathbf{\Omega}$.

# #### Calculate weight vector $\mathbf{W}$ in `W_df`  and FMP returns in `fmp_rt`

# - e.g:
#   - `fmp_rt['Growth']` contains Growth FMP returns.
#   - `W_df[list(W_df.keys())[-1]]['Inflation']` returns FMP weights for a macro factor, **Inflation**.

# In[28]:


start_test_dt = bf_test_trial.index[0]
W_df = {}
fmp_wt = pd.DataFrame()
fmp_rt = {}
for k in tqdm(B_df.keys()):
    B = B_df[k].to_numpy()
#     W = B@np.linalg.inv(B.T@B)
    W = B@np.linalg.pinv(B.T@B)    #  np.linalg.pinv() leverages SVD to approximate initial matrix, which might be used for almost singluar matrice.
    W_df[k] = pd.DataFrame(W, index=ds_bf.columns, columns=ds_mf.columns)
    temp_W = W_df[k].T.reset_index()
    temp_W['Date'] = k.end_time.strftime('%Y-%m-%d')
    fmp_wt = pd.concat((fmp_wt, temp_W))
    fmp_rt[k] = ds_bf.loc[k-1]@W_df[k]


# In[29]:


print('Step 2: FMP weights are calculated.')


# - `fmp_wt` is a big matrix that includes all FMP weights *over time*.

# In[30]:


fmp_wt = fmp_wt.reset_index(drop='True')
fmp_wt.rename(columns={'index':'Factor'}, inplace=True)
fmp_wt = fmp_wt[['Date'] + list(fmp_wt.columns[:-1])]


# - Save works.

# In[31]:


fmp_wt.to_csv('../../data/processed/fmp_wt.csv')
fmp_wt.to_pickle('../../data/processed/fmp_wt.pkl')


# In[32]:


fmp_wt.tail(10)


# # 4. Show model restuls.

# #### Calculate a macro factor return matrix `mf_rt`.

# - `mf_rt` is just values of macro factors as they are defined in `inputs.log`. We call them macro factor or macro factor returns, although they may not be really returns.
# - `fmp_cum[n]` is **n**-week cumulative returns of FMPs.

# In[33]:


fmp_rt = pd.DataFrame().from_dict(fmp_rt, orient='index')
scaler = StandardScaler()
mf_rt = X.loc[fmp_rt.index[0]:fmp_rt.index[-1], ds_mf.columns].copy()
fmp_rt.set_index(fmp_rt.index.to_timestamp(how='E').strftime('%Y-%m-%d'), inplace=True)
mf_rt.set_index(mf_rt.index.to_timestamp(how='E').strftime('%Y-%m-%d'), inplace=True)
lb_range = [1] + list(range(4,56,4))
fmp_cum = {cum:fmp_rt.rolling(window=cum).sum() for cum in lb_range}


# In[34]:


fmp_rt.tail()


# #### Compute correlation matrice between various-length FMP returns `cum` and factor returns.
# - Lookback-window for correlations is `window_sz`. e.g. Corr(52-week FMP returns, macro factor) when `window_sz`=52. 

# In[35]:


window_sz = [26, 52, 104]


# In[36]:


mf_len = len(ds_mf.columns)
r_corr = {}
cross_corr_rolling = None
print('Calculating correlations between FMP returns and macro factor returns...')

for cum in tqdm(fmp_cum.keys()):
    r_corr[cum] = {}
    for w in window_sz:
        r_corr[cum][w] = {}
        for r_idx in range(len(mf_rt.index)-w):
            c = pd.merge(fmp_cum[cum][r_idx:r_idx+w], mf_rt[r_idx:r_idx+w], how='inner', left_index=True, right_index=True, suffixes=('_fmp', '_mf')).corr()
            
            # We get a tuple of correlations(FMP returns, Macro factor returns)
            r_corr[cum][w][mf_rt.index[r_idx+w]] = tuple([c.iloc[i, i+mf_len] for i in range(mf_len)])  
        
        corr_fmp_mf = pd.DataFrame().from_dict(r_corr[cum][w], orient='index', columns=ds_mf.columns)
        if cross_corr_rolling is None:
            cross_corr_rolling = corr_fmp_mf.copy()
            cross_corr_rolling['lookback'] = cum
            cross_corr_rolling['window'] = w
        else:
            corr_fmp_mf['lookback'] = cum
            corr_fmp_mf['window'] = w
            cross_corr_rolling = pd.concat((cross_corr_rolling, corr_fmp_mf.copy()))

# Pull out the 'date' column from the index.
cross_corr_rolling.index.name='date'
cross_corr_rolling = cross_corr_rolling.reset_index()


# In[37]:


def make_corr_fig(n_week):
    mf_len = ds_mf.columns.shape[0]
    fig = plotly.subplots.make_subplots(rows=int(mf_len/2)+1, cols=2, horizontal_spacing = 0.05, vertical_spacing = 0.08,
                                       subplot_titles=['Correlation: '] + ds_mf.columns)
    
    for row, col_no in enumerate(range(0, len(ds_mf.columns),2)):
        fig.add_trace(
            go.Heatmap(
                name='{}'.format(ds_mf.columns[col_no]),
                z=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, ds_mf.columns[col_no]],
                x=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'date'],
                y=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'lookback'],
                colorscale='RdBu', zmin=-1, zmax=1
            ), row+1, 1)
        
        if col_no+1 < len(ds_mf.columns):
            fig.add_trace(
                go.Heatmap(
                    name='Corr. ({})'.format(ds_mf.columns[col_no+1]),
                    z=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, ds_mf.columns[col_no+1]],
                    x=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'date'],
                    y=cross_corr_rolling.loc[cross_corr_rolling.window==n_week, 'lookback'],
                    colorscale='RdBu', zmin=-1, zmax=1
                ), row+1, 2)

    fig['layout'].update(height=800, width=1200, title=str(n_week) + '-week correlations over time (y-week cumulative FMP return, macro factor)', template='plotly_white',
                         yaxis=dict(title='y-week'), autosize=False)

    return fig


# In[38]:


print('Making correlation figures...')


# ### `1) Rolling correlations`

# In[39]:


make_corr_fig(26).show()


# In[40]:


make_corr_fig(52).show()


# In[41]:


make_corr_fig(104).show()


# ### `2) FMP weights` (Spot)

# In[42]:


latest_dt = max(fmp_wt.Date)


# In[43]:


fig_fmp_wt = fmp_wt[fmp_wt.Date==latest_dt].drop(
    ['Date'],
    axis=1).set_index('Factor').iplot(asFigure=True,
                                      kind='barh',
                                      bargap=.2,
                                      colorscale='plotly',
                                      theme='white',
                                      title='매크로 팩터를 추종하는 포트폴리오의 자산구성 (예시)',
                                      yTitle='매크로 팩터',
                                      xTitle='비중',
                                      barmode='stack')
fig_fmp_wt['layout'].update(xaxis=dict(tickformat='.1%'),
                     yaxis=dict(categoryorder='category descending'),
                     legend_orientation='h')
fig_fmp_wt.layout.legend.title = '투자자산'


# In[44]:


iplot(fig_fmp_wt)


# In[45]:


fmp_wt_kor = fmp_wt.rename(
    columns={
        'DMEQ': '선진 주식',
        'UST': '미국 국채',
        'CRE': '투자등급 채권',
        'ILB': '미국 물가지수연동채권',
        'DXY': '달러지수',
        'FXCS': '자원부국 통화-안전자산 통화',
        'GOLD': '금',
        'ENGY': '에너지',
        'REIT': '리츠'
    }, inplace=False)


# In[46]:


fig_fmp_wt_kor = fmp_wt_kor[fmp_wt_kor.Date==latest_dt].drop(
    ['Date'],
    axis=1).set_index('Factor').iplot(asFigure=True,
                                      kind='barh',
                                      bargap=.2,
                                      colorscale='plotly',
                                      theme='white',
                                      title='매크로 팩터를 추종하는 포트폴리오의 자산구성 (예시)',
                                      yTitle='매크로 팩터',
                                      xTitle='비중',
                                      barmode='stack')
fig_fmp_wt_kor['layout'].update(xaxis=dict(tickformat='.1%'),
                     yaxis=dict(categoryorder='category descending'),
                     legend_orientation='h')
fig_fmp_wt_kor.layout.legend.title = '투자자산'


# In[47]:


iplot(fig_fmp_wt_kor)


# ### `3) FMP weights` (Time-series)

# #### Set `start_dt` and `end_dt` to plot.
# - Set *None* if you want to plot all available dates.
# - Format: yyyy-mm-dd

# In[212]:


start_dt = '2013-01-01'
end_dt = None


# In[213]:


min_dt = min(fmp_wt.Date) if start_dt is None else max(start_dt, min(fmp_wt.Date))
max_dt = max(fmp_wt.Date) if end_dt is None else min(end_dt, max(fmp_wt.Date))


# In[199]:


print('From {} to {}.'.format(min_dt, max_dt))


# - Set `shade_nm` to be phase names. Their corresponding dates will be shaded in the following FMP weights chart.

# In[200]:


shade_nm = ['Recovery', 'Expansion']
shade_colors=['gray', 'green']


# In[201]:





# In[202]:


legend_name = [' (Equities)', ' (UST 10yr)', ' (BBB-AAA)', ' (TIPS)', ' (Dollar Index)', ' (FX:Commidity-Safe)', '', ' (Energy)', '']


# #### Set `plot_name` to be a phase name to be plotted.

# In[251]:


print('Macro factors are: {}'.format(ds_mf.columns))


# In[204]:


plot_name = 'Growth'


# #### Set `visible_base_assets` to be base asset names visible, which may be turned on and off later.

# In[192]:


visible_base_assets = ['DMEQ', 'CRE', 'GOLD']


# In[214]:


def make_fmp_wt_fig(plot_name='Growth',
                    shade_nm=['Recovery', 'Expansion'],
                    shade_colors=['gray', 'green'],
                    start_dt=None,
                    end_dt=None,
                    visible_base_assets=['DMEQ', 'CRE', 'GOLD']):

    # Set start date and end date to plot.
    min_dt = min(fmp_wt.Date) if start_dt is None else max(start_dt, min(fmp_wt.Date))
    max_dt = max(fmp_wt.Date) if end_dt is None else min(end_dt, max(fmp_wt.Date))
    
    # Set dates for shading.
    shade1_dt = pd.Series(phase[phase==shade_nm[0]].index.to_timestamp(), name='date')
    shade2_dt = pd.Series(phase[phase==shade_nm[1]].index.to_timestamp(), name='date')
    shade1_start, shade1_end = get_start_end_dates(shade1_dt, 'W')
    shade2_start, shade2_end = get_start_end_dates(shade2_dt, 'W')
    
    # Set a figure.
    fig_fmp_wt_ts = fmp_wt[fmp_wt.Factor==plot_name].loc[np.logical_and(fmp_wt.Date>=min_dt, fmp_wt.Date<=max_dt), ~fmp_wt.columns.isin(['Factor'])].iplot(
        asFigure=True,
        kind='bar',
        x='Date',
        colorscale='plotly',
        barmode='stack',
        theme='white',
        world_readable=True,
        title='Changes in {} Factor-Mimicking-Portfolio weights over time'.format(plot_name),
        xTitle='Date',
        yTitle='Weights'
    )

    # Draw shades.
    fig_fmp_wt_ts = plotly_multi_shades(fig_fmp_wt_ts, x0=[shade1_start, shade2_start], x1=[shade1_end, shade2_end], colors=shade_colors, alpha=0.2)
    fig_fmp_wt_ts['layout'].update(yaxis=dict(tickformat='%'))
    fig_fmp_wt_ts.layout.legend.title = 'Base assets'
    fig_fmp_wt_ts = fig_fmp_wt_ts.update_xaxes(range=[min_dt, max_dt])

    # Set visible base assets and make legends with more explanation as needed.
    for i, fdata in enumerate(fig_fmp_wt_ts.data):
        fdata.visible = 'legendonly' if fig_fmp_wt_ts.data[i].name not in visible_base_assets else None
        fdata.name = fig_fmp_wt_ts.data[i].name + legend_name[i]

    fig_fmp_wt_ts.update_layout(
        annotations=[
            dict(
                x=0,
                y=0,
                xref='paper',
                yref='paper',
                text="Gray: {} phase, Green: {} phase".format(shade_nm[0], shade_nm[1]),
                showarrow=False
            )
        ]
    );
    
    return fig_fmp_wt_ts


# In[211]:


fmp_wt[fmp_wt.Factor==plot_name].loc[np.logical_and(fmp_wt.Date>=min_dt, fmp_wt.Date<=max_dt), ~fmp_wt.columns.isin(['Factor'])]


# In[195]:


fig_fmp_wt_ts = fmp_wt[fmp_wt.Factor==plot_name].loc[np.logical_and(fmp_wt.Date>=min_dt, fmp_wt.Date<=max_dt), ~fmp_wt.columns.isin(['Factor'])].iplot(
    asFigure=True,
    kind='bar',
    x='Date',
    colorscale='plotly',
    barmode='stack',
    theme='white',
    world_readable=True,
    title='Changes in {} Factor-Mimicking-Portfolio weights over time'.format(plot_name),
    xTitle='Date',
    yTitle='Weights'
)
fig_fmp_wt_ts = plotly_multi_shades(fig_fmp_wt_ts, x0=[shade1_start, shade2_start], x1=[shade1_end, shade2_end], colors=['gray', 'green'], alpha=0.2)
fig_fmp_wt_ts['layout'].update(yaxis=dict(tickformat='%'))
fig_fmp_wt_ts.layout.legend.title = 'Base assets'
fig_fmp_wt_ts = fig_fmp_wt_ts.update_xaxes(range=[min_dt, max_dt])

for i, fdata in enumerate(fig_fmp_wt_ts.data):
    fdata.visible = 'legendonly' if fig_fmp_wt_ts.data[i].name not in visible_base_assets else None
    fdata.name = fig_fmp_wt_ts.data[i].name + legend_name[i]

fig_fmp_wt_ts.update_layout(
    annotations=[
        dict(
            x=0,
            y=0,
            xref='paper',
            yref='paper',
            text="Gray: {} phase, Green: {} phase".format(shade_nm[0], shade_nm[1]),
            showarrow=False
        )
    ]
);


# In[256]:


fig = make_fmp_wt_fig(plot_name=plot_name,
                      shade_nm=shade_nm,
                      shade_colors=shade_colors,
                      start_dt=None,
                      end_dt=None,
                      visible_base_assets=visible_base_assets)


# In[257]:


iplot(fig)


# # 5. Save works

# #### 1. Tables (csv files)
# - `results_to_save` is a list of instances to be saved in .pkl and .csv, where their file names are specified in `filenames` in order.
# - `correlations`: Cross correlations between `n`-week cumulative FMP returns and macro factors.
# - `fmp_rt`: FMP returns. Return frequency is `_freq`.
# - `macrofactor_rt`: Macro factor (returns).
# - `fmp_wt`: Macro factor weights. **FILAL RESULTS of this model.**
# 
# #### 2. Charts (png, pdf, json files)
# - `results_to_save` is a list of instances to be saved in .png, .pdf and .json, where their file names are specified in `filenames` in order.
# - `corr_w##`: Rolling correlation charts, where their lookback window size is ##.

# In[254]:


def save_works(what='all'):
    '''
    Save results.
    
    Parameters:
        what: {'all'|'tables'|'figures'}
    '''
    
    today = date.today().strftime('%Y-%m-%d')
    
    # Save tables
    if (what == 'all') or (what == 'tables'):
        results_to_save = [cross_corr_rolling, fmp_rt, mf_rt, fmp_wt]
        filenames = ['/correlations', '/fmp_rt', '/macrofactor_rt', '/fmp_wt']
        assert len(results_to_save) == len(filenames), 'The number of elements in both lists, results_to_save and filenames, must be equal.'
        
        table_path = '../../reports/tables/'
        paths = [table_path + today + filename for filename in filenames]

        # Make a folder if needed.
        if not os.path.exists(table_path+today):
            os.mkdir(table_path+today)

        # Get sequential file names.
        filename_seq_pkl = [get_nonexistant_path(path+'.pkl') for path in paths]
        filename_seq_csv = [get_nonexistant_path(path+'.csv') for path in paths]
        
        # We save the same results in tables in different formats: csv and pickle.
        for idx in range(len(paths)):
            results_to_save[idx].to_pickle(filename_seq_pkl[idx])
            results_to_save[idx].to_csv(filename_seq_csv[idx])
        
        print('Tables are saved in {}.'.format(table_path))
    
    # Save charts.
    if (what == 'all') or (what == 'figures'):
        results_to_save = [make_corr_fig(26), make_corr_fig(52), make_corr_fig(104), fig_fmp_wt, fig_fmp_wt_kor] + [make_fmp_wt_fig(plot_name=mf) for mf in ds_mf.columns]
        filenames = ['/corr_w26', '/corr_w52', '/corr_w104', '/fmp_wt', '/fmp_wt_kor'] + ['/fmp_wt_ts_' + mf for mf in ds_mf.columns]
        assert len(results_to_save) == len(filenames), 'The number of elements in both lists, results_to_save and filenames, must be equal.'

        figure_path = '../../reports/figures/'
        paths = [figure_path + today + filename for filename in filenames]

        # Make a folder if needed.
        if not os.path.exists(figure_path+today):
            os.mkdir(figure_path+today)

        # Get sequential file names.
        filename_seq_png = [get_nonexistant_path(path+'.png') for path in paths]
        filename_seq_pdf = [get_nonexistant_path(path+'.pdf') for path in paths]
        filename_seq_json = [get_nonexistant_path(path+'.json') for path in paths]
       
        # We save the same charts in different formats: png, svg and json.
        for idx in range(len(paths)):
            results_to_save[idx].write_image(filename_seq_png[idx])
            results_to_save[idx].write_image(filename_seq_pdf[idx])
            plotly_fig2json(results_to_save[idx], filename_seq_json[idx])
        
        print('Charts are saved in {}.'.format(figure_path))


# In[255]:


save_works()


# In[ ]:





# In[ ]:




