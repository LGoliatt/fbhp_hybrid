#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import pygmo as pg

#%%----------------------------------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings('ignore')

import sys, getopt
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

#print ("This is the name of the script: ", program_name)
#print ("Number of arguments: ", len(arguments))
#print ("The arguments are: " , arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0
else:
    run0, n_runs = 0,50
#%%----------------------------------------------------------------------------   
def accuracy_log(y_true, y_pred):
    """
Calculates the percentage of predictions within a logarithmic error range.

    Args:
        y_true: The ground truth values.
        y_pred: The predicted values.

    Returns:
        float: The accuracy score as a percentage, representing the proportion of 
               predictions where the absolute value of the base-10 logarithm of 
               the ratio between true and predicted values is less than 0.3.
    """
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100

def rms(y_true, y_pred):
    """
Calculates the Root Mean Square (RMS) error between two arrays.

    Args:
        y_true: The array of true values.
        y_pred: The array of predicted values.

    Returns:
        float: The RMS error value.
    """
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return ( (np.log10(y_pred/y_true)**2).sum()/len(y_true) )**0.5

def model_base_evaluation(x, data_args, estimator_args,):
    
  """
Evaluates or runs a model based on the provided flags.

    Args:
        x: Model parameters.
        data_args: Data-related arguments including train/test splits and scoring metrics.
        estimator_args: Estimator-related arguments including name and parameters.

    Returns:
        None: If flag is 'eval'.
        dict: A dictionary containing evaluation results if flag is 'run'.  The dictionary includes true and predicted values for training and testing sets, estimator parameters, model parameters, the active variables, seed, number of splits, and output target.
    """
    
  (X_train_, y_train, X_test_, y_test, flag, task,  n_splits, 
     random_seed, scoring, target, 
     n_samples_train, n_samples_test, n_features)   = data_args
  (clf_name, n_decision_variables, p, clf)          = estimator_args  #
  
  if flag=='eval':

        return None
    
  elif flag=='run':
        
    return {
            'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':y_p, 
            'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_t,             
            'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':clfnme,
            #'SCALES_PARAMS':{'scaler':n},
            #'TRANSF_PARAMS':{'tranformer':t, 'kernel':k, 'n_components':n_components},
            #'ESTIMATOR':clf, 
            'ACTIVE_VAR':ft, #'SCALER':n,
            'SEED':random_seed, 'N_SPLITS':n_splits,
            #'ACTIVE_FEATURES':ft,
            'OUTPUT':target
            }
  else:
      sys.exit('Model evaluation doe not performed for estimator '+clf_name)
      
#------------------------------------------------------------------------------
#%%----------------------------------------------------------------------------     
def fun_xgb_fs(x,*data_args):
  
  """
Evaluates an XGBoost model with hyperparameters defined by the input vector.

  Args:
    x: A vector containing the hyperparameters for the XGBoost model 
       (learning_rate, n_estimators, max_depth, reg_lambda).
    data_args: Variable length argument list containing training and testing data, 
               flags, task details, cross-validation parameters, random seed, 
               scoring metric, target variable, and sample sizes.

  Returns:
    None: The function returns the output of `model_base_evaluation`.
  """
  
  #print(data_args,'1>>')  
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='XGB' 
  n_decision_variables  = 4
    
  cr ={0:'reg:squarederror', 1:'reg:logistic', 2:'binary:logistic',}
  clf = XGBRegressor(random_state=int(random_seed), objective=cr[0], n_jobs=1)
  p={
     'learning_rate'        : int(x[0]*1000)/1000.,
     'n_estimators'         : int(x[1]+0.99), 
     'max_depth'            : int(x[2]+0.99),
     'reg_lambda'           : int(x[3]*1000)/1000.,
    
     }
    
  clf.set_params(**p)  
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, )
#%%----------------------------------------------------------------------------     
def fun_svr_fs(x,*data_args):
  """
Evaluates an SVR model with specified parameters.

  Args:
    x: Optimization variables for gamma, C, and epsilon.
    data_args: A tuple containing training/testing data, flags, task details, 
               cross-validation settings, scoring metrics, target variable information,
               and dataset sizes/features.

  Returns:
    None: The function returns the output of `model_base_evaluation`.
  """
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='SVR' 
  n_decision_variables  = 3
  
  clf = SVR(kernel='rbf', max_iter=1000)
  kernel = {
            0:'rbf', 
            1:'sigmoid', 
            2:'chi2',
            3:'poly', 
            4:'linear', 
            5:'laplacian', 
            }  
  
  _gamma = x[0]
  p={
     'gamma'        :'scale' if _gamma<=0 else _gamma, 
     'C'            : x[1],  
     'epsilon'      : x[2], 
     }

  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, )
#%%----------------------------------------------------------------------------     
def fun_elm_fs(x,*data_args):
  """
Evaluates an Extreme Learning Machine (ELM) regressor with specified parameters.

  Args:
    x: A list or array containing the optimization variables for ELM.
    data_args: Tuple of arguments including training and testing data, flags, 
               task details, cross-validation settings, random seed, scoring metric,
               target variable, and dataset sizes/features.

  Returns:
    None: The function returns the output of `model_base_evaluation`, which is not explicitly defined here but presumably contains evaluation results.
  """
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='ELM' 
  n_decision_variables  = 5
  
  clf = ELMRegressor(random_state=int(random_seed))
  af = {
      0 :'identity', 
      4 :'relu', 
      5 :'swish',
      1 :'gaussian', 
      2 :'multiquadric', 
      3 :'inv_multiquadric',
  }

  _alpha=int(x[4]*1000)/1000.
  regressor = None if _alpha<1e-4 else Ridge(alpha=_alpha,random_state=int(random_seed))
  m=1e4
  p={'n_hidden'         : int(round(x[0])), #'alpha':1, 'rbf_width':1,
     'activation_func'  : af[int(x[1]+0.995)], #'alpha':0.5, 
     'alpha'            : int(x[2]*m)/m, 
     'rbf_width'        : int(x[3]*m)/m,
     'regressor'        : regressor,
     }
  clf.set_params(**p)
  p['l2_penalty']=_alpha
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, )
#%%---------------------------------------------------------------------------- 

#------------------------------------------------------------------------------
def fun_mars_fs(x,*data_args):
  """
Trains and evaluates a MARS model with specified parameters.

  Args:
    x: A list or array containing the hyperparameters for the MARS model 
       (max_degree, penalty, max_terms).
    data_args: Tuple of arguments to be passed to the evaluation function.
               Includes training and testing data, flags, task details, 
               and other relevant parameters.

  Returns:
    None: The method returns the output of `model_base_evaluation`.
  """
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='MARS'
  n_decision_variables  = 3
  
  clf = MARS()
  p={
   'allow_linear'             : True, 
   'allow_missing'            : False, 
   'check_every'              : 1,
   'enable_pruning'           : True, 
   'endspan'                  : None, 
   'endspan_alpha'            : None, 
   'fast_K'                   : 5,
   'fast_h'                   : 1, 
   'feature_importance_type'  : None, 
   'max_degree'               : np.round(x[0]),
   'max_terms'                : 1000, 
   'min_search_points'        : 100, 
   'minspan'                  : None,
   'minspan_alpha'            : None, 
   'penalty'                  : x[1], 
   'smooth'                   : False, 
   'thresh'                   : 0.001,  
   'use_fast'                 : True, 
   'verbose'                  : 0, 
   'zero_tol'                 : 1e-6,
   #'rcond'                    : -1,
    }
  clf.set_params(**p)
  p={
   'max_degree'               : np.round(x[0]),
   'penalty'                  : x[1],
   'max_terms'                : int(round(x[2])),
    }
  
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, )
#------------------------------------------------------------------------------


#%%----------------------------------------------------------------------------   
from scipy.optimize import differential_evolution as de
import glob as gl
import pylab as pl
import os

basename='de_'

#%%
from read_data_bottom_hole import *
datasets = [
                read_ayoub(),
            ]
#%%----------------------------------------------------------------------------   
pd.options.display.float_format = '{:.3f}'.format

pop_size    = 30
max_iter    = 50
n_splits    = 5
scoring     = 'neg_mean_squared_error'
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run+1001
    
    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+basename+dr+'/'
        print(path)
        
        os.system('mkdir  '+path)
        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            dataset_name = dataset['name']+'-'+tn
            target                          = dataset['target_names'][tk]
            y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            X_train, X_test                 = dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            
            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Output                     : '+tn+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            #s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'            
            
            print(s)
            e=1e-5
            #------------------------------------------------
            lb_elm = [  1e-0,    0,    1,   1., 0.0] #+ [0.0]*n_features
            ub_elm = [  3e+2,    6,  1+e,  10., 1e4] #+ [1.0]*n_features
            #------------------------------------------------         
            lb_svr = [     0, 1e-1, 1e-1,        ] #+ [0.0]*n_features
            ub_svr = [  1e+1, 1e+4, 1e+2,        ] #+ [1.0]*n_features
            #------------------------------------------------         
            lb_xgb = [  1e-6,   10,    1,   0.,  ] #+ [0.0]*n_features
            ub_xgb = [  1e+0,  100,   20, 100.,  ] #+ [1.0]*n_features
            #------------------------------------------------         
            lb_mars= [     0,    1,    1,        ] #+ [0.0]*n_features
            ub_mars= [     2,  10,  100,         ] #+ [1.0]*n_features
            #------------------------------------------------         
            args = (X_train, y_train, X_test, y_test, 'eval', task,  n_splits, 
                    int(random_seed), scoring, target, 
                    n_samples_train, n_samples_test, n_features,)
            #------------------------------------------------------------------         
            estimators=[             
                ('ELM'  , lb_elm, ub_elm, fun_elm_fs, args, random_seed,),    # OK
                ('SVR'  , lb_svr, ub_svr, fun_svr_fs, args, random_seed,),    # OK
                ('XGB'  , lb_xgb, ub_xgb, fun_xgb_fs, args, random_seed,),    # OK
                ('MARS' ,lb_mars,ub_mars,fun_mars_fs, args, random_seed,),    # OK
            ]
            #------------------------------------------------------------------         
            for (clf_name, lb, ub, fun, args, random_seed) in estimators:
                np.random.seed(random_seed)
                list_results=[]
                #--------------------------------------------------------------
                s=''
                s='-'*80+'\n'
                s+='Estimator                  : '+clf_name+'\n'
                s+='Function                   : '+str(fun)+'\n'
                s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
                s+='Output                     : '+tn+'\n'
                s+='Run                        : '+str(run)+'\n'
                s+='Random seed                : '+str(random_seed)+'\n'

                print(s)                
##%%----------------------------------------------------------------------------
