#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import glob as gl
import pylab as pl
import pandas as pd
import os


#%%
def read_ayoub(
                            plot=False, 
                            expand_features=False, 
                            var_set=None,
                            ):
    """
Reads and processes the Ayoub dataset.

    This method reads data from a JSON file, splits it into training and testing sets,
    performs descriptive statistics, generates correlation heatmaps, and prepares
    the data for regression tasks.

    Args:
        plot (bool, optional):  Whether to generate plots. Defaults to False.
        expand_features (bool, optional): Whether to expand features. Defaults to False.
        var_set (int, optional): Specifies a subset of variables to use. 
                                  Defaults to None.

    Returns:
        dict: A dictionary containing the processed regression data, including training and testing sets,
              feature names, target names, and other relevant information.
    """
    #%% 
    filename='./data/ayoub/ayoub.json'
    data = pd.read_json(filename)

    df_test = data[data['dataset']=='test']
    df_test.drop('dataset', axis=1, inplace=True)

    d1_train = data[data['dataset']=='train']
    d1_train.drop('dataset', axis=1, inplace=True)

    d2_train = data[data['dataset']=='validate']
    d2_train.drop('dataset', axis=1, inplace=True)

    df_train = pd.concat([d1_train, d2_train], axis=0)

    target_names=['MBHP']
    aux=df_test[target_names].values
    df_test.drop(target_names, axis=1, inplace=True)
    df_test[target_names]=aux
    aux=df_train[target_names].values
    df_train.drop(target_names, axis=1, inplace=True)
    df_train[target_names]=aux
    
    pd.options.display.float_format = '{:.2f}'.format
    for d,df in [('train', df_train), ('test',df_test)]:
        #print(d)
        desc=df.describe(percentiles=[0.5]).T
        desc=df.describe().T
        print(desc, '\n\n\n')
        desc.to_excel(d+'.xlsx')
        desc.to_latex(d+'.tex', index=True)
        
    #%%
    pl.rc('text', usetex=True)
    pl.rc('font', family='serif',  serif='Times')
    import seaborn as sns
    df=df_train.copy(); 
    corr = df.corr()
    corr.to_excel('corr.xlsx')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = pl.subplots(figsize=(7, 6))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap =  cmap="YlGnBu"
    g=sns.heatmap(corr, mask=mask, cmap=cmap, #vmax=.3, center=0, 
            annot=True, fmt='.2g', #fmt="d")
            square=True, linewidths=.5, #cbar_kws={"shrink": .5},
        )
    g.set_yticklabels(g.get_yticklabels(), rotation = 0,)# fontsize = 8)
    pl.show()
    #%%
    variable_names=df_train.columns.drop(target_names)  

    if var_set==0:
        id_var_set=[2,3,4,8,0] # MARS
    if var_set==1:
        id_var_set=[2,3,4,8,] # MARS
    elif var_set==3:
        id_var_set=[2,3,4,8,0] # MARS
    elif var_set==2:
        id_var_set=[2,4,8,0] # XGB
    elif var_set==4:
        id_var_set=[2,3,4,8,0,7] # MARS
    elif var_set==5:
        id_var_set=[2,3,4,8,0,1] # MARS
    else:
        id_var_set=range(len(variable_names))


    X_train, X_test = df_train[variable_names].values, df_test[variable_names].values    
    y_train, y_test = df_train[target_names].values, df_test[target_names].values    
    n_samples, n_features = X_train.shape 


    suffix='' if var_set==None else' V'+str(var_set)
    regression_data =  {
      'task'            : 'regression',
      'name'            : 'ayoub'+suffix,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "",
      'normalize'       : 'None',
      'stations'        : variable_names,
      'date_range'      : None,
      }
    #%%
    return regression_data
    #%%
    
if __name__ == "__main__":
    datasets = [                 
            read_ayoub(),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
    
