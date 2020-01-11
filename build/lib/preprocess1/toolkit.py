import pandas as pd
import numpy as np
from scipy import stats # for mode imputation
from fastai.tabular import add_datepart # fot Time FE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest # for outlier detection
#import featuretools as ft # for automated FE
import ipywidgets as wg 
from IPython.display import display
from ipywidgets import Layout
from sklearn import preprocessing as pre
from scipy import stats
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import power_transform
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import cluster
# from pyod.models.knn import KNN
# from pyod.models.iforest import IForest
# from pyod.models.pca import PCA as PCA_od
# from pyod.models.hbos import HBOS
import sys 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
import datefinder
from datetime import datetime
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set()
#_____________________________________________________________________________________________________________________________

class DataTypes_Auto_infer(BaseEstimator,TransformerMixin):
  '''
    - This will try to infer data types automatically, option to override learent data types is also available.
    - This alos automatically delets duplicate columns (values or same colume name), removes rows where target variable is null and 
      remove columns and rows where all the records are null
  '''

  def __init__(self,target,ml_usecase,catagorical_features=[],numerical_features=[],time_features=[]): # nothing to define
    '''
    User to define the target (y) variable
      args:
        target: string, name of the target variable
        ml_usecase: string , 'regresson' or 'classification . For now, only supports two  class classification
        - this is useful in case target variable is an object / string . it will replace the strings with integers
        catagorical_features: list of catagorical features, default None, when None best guess will be used to identify catagorical features
        numerical_features: list of numerical features, default None, when None best guess will be used to identify numerical features
        time_features: list of date/time features, default None, when None best guess will be used to identify date/time features    
  '''
    self.target = target
    self.ml_usecase= ml_usecase
    self.catagorical_features =catagorical_features
    self.numerical_features = numerical_features
    self.time_features =time_features
  
  def fit(self,data,y=None): # learning data types of all the columns
    '''
    Args: 
      data: accepts a pandas data frame
    Returns:
      Panda Data Frame
    '''
    # we will take float as numberic, object as catagorical from the begning
    # fir int64, we will check to see what is the proportion of unique counts to the total lenght of the data
    # if proportion is lower, then it is probabaly catagorical 
    # however, proportion can be lower / disturebed due to samller denominator (total lenghth / number of samples)
    # so we will take the following chart
    # 0-50 samples, threshold is 24%
    # 50-100 samples, th is 12%
    # 50-250 samples , th is 4.8%
    # 250-500 samples, th is 2.4%
    # 500 and above 2% or belwo

    # we canc check if somehow everything is object, we can try converting them in float
    for i in data.select_dtypes(include=['object']).columns:
      try:
        data[i] = data[i].astype('int64')
      except:
        None

    # should really be an absolute number say 20
    # length = len(data.iloc[:,0])
    # if length in range(0,51):
    #   th=.25
    # elif length in range(51,101):
    #   th=.12
    # elif length in range(101,251):
    #   th=.048
    # elif length in range(251,501):
    #   th=.024
    # elif length > 500:
    #   th=.02

    # if column is int and unique counts are more than two, then: (exclude target)
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtypes == 'int64': #((data[i].dtypes == 'int64') & (len(data[i].unique())>2))
        if len(data[i].unique())<=20: #hard coded
          data[i]= data[i].apply(str)
        else:
          data[i]= data[i].astype('float64')
    
    # # if colum is object and only have two unique counts , this is probabaly one hot encoded
    # # make it int64
    # for i in data.columns:
    #   if ((data[i].dtypes == 'object') & (len(data[i].unique())==2)):
    #     unique_cat = list(data[i].unique())
    #     data[i].replace(unique_cat,[0,1],inplace=True)
    
    #for time & dates
    for i in data.drop(self.target,axis=1).columns:
      # we are going to check every first row of every column and see if it is a date
      match = datefinder.find_dates(data.loc[0,i])
      try:
        for m in match:
          if isinstance(m, datetime) == True:
            data[i] = pd.to_datetime(data[i])
      except:
        continue

    # now in case we were given any specific columns dtypes in advance , we will over ride theos 
    if len(self.catagorical_features) > 0:
      for i in self.catagorical_features:
        data[i]=data[i].apply(str)
    
    if len(self.numerical_features) > 0:
      for i in self.numerical_features:
        data[i]=data[i].astype('float64')
    
    if len(self.time_features) > 0:
      for i in self.time_features:
        data[i]=pd.to_datetime(data[i])

    # table of learent types
    self.learent_dtypes = data.dtypes
    self.training_columns = data.drop(self.target,axis=1).columns

    # lets remove duplicates
    # remove duplicate columns (columns with same values)
    data_c = data.T.drop_duplicates()
    data = data_c.T
    #remove columns with duplicate name 
    data = data.loc[:,~data.columns.duplicated()]
    # Remove NAs
    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    # remove the row if target column has NA
    data[self.target].dropna(axis=0, how='all', inplace=True)
    self.training_columns = data.drop(self.target,axis=1).columns

    # since due to transpose , all data types have changed, lets change the dtypes to original
    for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
      data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    
    # now make the interface and ask for user types, we are ignoring target, since we already got the information from user already
    # col = len(self.learent_dtypes.drop(index = self.target).index)
    # self.dic = {}
    # display(wg.Text(value="We identified following data types, change them as you may please , Hit Enter when done",layout =Layout(width='50%')))
    # for i,c in zip(range(col),self.learent_dtypes.drop(index = self.target).index):
    #   self.dic["inpt_"+ str(i)] = wg.Dropdown(options=["float64","object","datetime64"],value=str(self.learent_dtypes[i]),description=self.learent_dtypes.index[i])
    #   display(self.dic["inpt_"+ str(i)])
    # input()
    
    display(wg.Text(value="We identified following data types,if they are incorrect , enter ''quit'' , and provide correct types in the argument",layout =Layout(width='50%')))
    display(pd.DataFrame(self.learent_dtypes, columns=['Feature_Type']))
    response = input()

    if response in ['quit','Quit','exit','EXIT','q','Q','e','E']:
      sys.exit()
    
    return(data)
  
  def transform(self,data,y=None):
    '''
      Args: 
        data: accepts a pandas data frame
      Returns:
        Panda Data Frame
    '''
    #very first thing we need to so is to check if the training and test data hace same columns
    #exception checking   
    import sys
    
    #checking train size parameter
    if sum(self.training_columns.sort_values() !=data.columns.sort_values())>0:
        sys.exit('(Type Error): train and test dataset does not have same columns.')

    # only take columns that training has

    data = data[self.training_columns] 
    # just keep picking the data and keep applying to the test data set (be mindful of target variable)
    for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
      data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    

    return(data)

  # fit_transform
  def fit_transform(self,data,y=None):
    # since this is for training , we dont nees any transformation since it has already been transformed in fit
    data = self.fit(data)

    # additionally we just need to treat the target variable
    # for ml use ase
    if ((self.ml_usecase == 'classification') &  (data[self.target].dtype=='object')):
      self.u = list(pd.unique(data[self.target]))
      self.replacement = np.arange(0,len(self.u))
      data[self.target]= data[self.target].replace(self.u,self.replacement)
      data[self.target] = data[self.target].astype('int64')
      self.replacement = pd.DataFrame(dict(target_variable=self.u,replaced_with=self.replacement))
    
    
    
    return(data)
# _______________________________________________________________________________________________________________________
# Imputation
class Simple_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes all type of data (numerical,catagorical & Time).
      Highly recommended to run Define_dataTypes class first
      Numerical values can be imputed with mean or median 
      Catagorical missing values will be replaced with "Other"
      Time values are imputed with the most frequesnt value
      Ignores target (y) variable    
      Args: 
        Numeric_strategy: string , all possible values {'mean','median'}
        catagorical_strategy: string , all possible values {'not_available','most frequent'}
        target: string , name of the target variable

  '''

  def __init__(self,numeric_strategy,catagorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.catagorical_strategy = catagorical_strategy
  
  def fit(self,data,y=None): #
    # make a table for numerical variable with strategy stats
    if self.numeric_strategy == 'mean':
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmean)
    else:
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmedian)

    self.numeric_columns = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).columns

    #for Catgorical , 
    if self.catagorical_strategy == 'most frequent':
      self.catagorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.catagorical_stats = pd.DataFrame(columns=self.catagorical_columns) # place holder
      for i in (self.catagorical_stats.columns):
        self.catagorical_stats.loc[0,i] = data[i].value_counts().index[0]
    else:
      self.catagorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
    
    # for time, there is only one way, pick up the most frequent one
    self.time_columns = data.drop(self.target,axis=1).select_dtypes(include=['datetime64[ns]']).columns
    self.time_stats = pd.DataFrame(columns=self.time_columns) # place holder
    for i in (self.time_columns):
      self.time_stats.loc[0,i] = data[i].value_counts().index[0]
    return(data)

      
  
  def transform(self,data,y=None): 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      data[i].fillna(s,inplace=True)
    
    # for catagorical columns
    if self.catagorical_strategy == 'most frequent':
      for i in (self.catagorical_stats.columns):
        #data[i].fillna(self.catagorical_stats.loc[0,i],inplace=True)
        data[i] = data[i].fillna(self.catagorical_stats.loc[0,i])
        data[i] = data[i].apply(str)
    else: # this means replace na with "not_available"
      for i in (self.catagorical_columns):
        data[i].fillna("not_available",inplace=True)
        data[i] = data[i].apply(str)
    # for time
    for i in (self.time_stats.columns):
        data[i].fillna(self.time_stats.loc[0,i],inplace=True)
    
    return(data)
  
  def fit_transform(self,data,y=None):
    data= self.fit(data)
    return(self.transform(data))

# _______________________________________________________________________________________________________________________
# Imputation with surrogate columns
class Surrogate_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes feature with surrogate column (numerical,catagorical & Time).
      - Highly recommended to run Define_dataTypes class first
      - it is also recommended to only apply this to features where it makes business sense to creat surrogate column
      - feature name has to be provided
      - only able to handle one feature at a time
      - Numerical values can be imputed with mean or median 
      - Catagorical missing values will be replaced with "Other"
      - Time values are imputed with the most frequesnt value
      - Ignores target (y) variable    
      Args: 
        feature_name: string, provide features name
        feature_type: string , all possible values {'numeric','catagorical','date'}
        strategy: string ,all possible values {'mean','median','not_available','most frequent'}
        target: string , name of the target variable

  '''
  def __init__(self,numeric_strategy,catagorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.catagorical_strategy = catagorical_strategy
  
  def fit(self,data,y=None): #
    # make a table for numerical variable with strategy stats
    if self.numeric_strategy == 'mean':
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmean)
    else:
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmedian)

    self.numeric_columns = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).columns
    # also need to learn if any columns had NA in training
    self.numeric_na = pd.DataFrame(columns=self.numeric_columns)
    for i in self.numeric_columns:
      if data[i].isna().any() == True:
        self.numeric_na.loc[0,i] = True
      else:
        self.numeric_na.loc[0,i] = False 

    #for Catgorical , 
    if self.catagorical_strategy == 'most frequent':
      self.catagorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.catagorical_stats = pd.DataFrame(columns=self.catagorical_columns) # place holder
      for i in (self.catagorical_stats.columns):
        self.catagorical_stats.loc[0,i] = data[i].value_counts().index[0]
      # also need to learn if any columns had NA in training, but this is only valid if strategy is "most frequent"
      self.catagorical_na = pd.DataFrame(columns=self.catagorical_columns)
      for i in self.catagorical_columns:
        if data[i].isna().any() == True:
          self.catagorical_na.loc[0,i] = True
        else:
          self.catagorical_na.loc[0,i] = False        
    else:
      self.catagorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.catagorical_na = pd.DataFrame(columns=self.catagorical_columns)
      self.catagorical_na.loc[0,:] = False #(in this situation we are not making any surrogate column)
    
    # for time, there is only one way, pick up the most frequent one
    self.time_columns = data.drop(self.target,axis=1).select_dtypes(include=['datetime64[ns]']).columns
    self.time_stats = pd.DataFrame(columns=self.time_columns) # place holder
    self.time_na = pd.DataFrame(columns=self.time_columns)
    for i in (self.time_columns):
      self.time_stats.loc[0,i] = data[i].value_counts().index[0]
    
    # learn if time columns were NA
    for i in self.time_columns:
      if data[i].isna().any() == True:
        self.time_na.loc[0,i] = True
      else:
        self.time_na.loc[0,i] = False
    
    return(data) # nothing to return

      
  
  def transform(self,data,y=None): 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      array = data[i].isna()
      data[i].fillna(s,inplace=True)
      # make a surrogate column if there was any
      if self.numeric_na.loc[0,i] == True:
        data[i+"_surrogate"]= array
        # make it string
        data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)

    
    # for catagorical columns
    if self.catagorical_strategy == 'most frequent':
      for i in (self.catagorical_stats.columns):
        #data[i].fillna(self.catagorical_stats.loc[0,i],inplace=True)
        array = data[i].isna()
        data[i] = data[i].fillna(self.catagorical_stats.loc[0,i])
        data[i] = data[i].apply(str)
        # make surrogate column
        if self.catagorical_na.loc[0,i] == True:
          data[i+"_surrogate"]= array
          # make it string
          data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)
    else: # this means replace na with "not_available"
      for i in (self.catagorical_columns):
        data[i].fillna("not_available",inplace=True)
        data[i] = data[i].apply(str)
        # no need to make surrogate since not_available is itself a new colum
    
    # for time
    for i in (self.time_stats.columns):
      array = data[i].isna()
      data[i].fillna(self.time_stats.loc[0,i],inplace=True)
      # make surrogate column
      if self.time_na.loc[0,i] == True:
        data[i+"_surrogate"]= array
        # make it string
        data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)
    
    return(data)
  
  def fit_transform(self,data,y=None):
    data= self.fit(data)
    return(self.transform(data))

 

# _______________________________________________________________________________________________________________________
# _______________________________________________________________________________________________________________________
# Time Feature Extraction
class Make_Time_Features(BaseEstimator,TransformerMixin):
  '''
    -Given a time feature , it extracts more features
    - full list of features is:
      ['Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed']
    - all extracted features are defined as string / object
    -it is recommended to run Define_dataTypes first
      Args: 
        time_feature: string , name of the Date variable as datetime64[ns]
        list_of_features: list of required features , default value ['Month','Dayofweek']

  '''

  def __init__(self,time_feature=[],list_of_features=['Month','Dayofweek']):
    self.time_feature = time_feature
    self.list_of_features_o = list_of_features
  
  def fit(self,data,y=None): # nothing to learn,
    return(None)
  
  def transform(self,data,y=None): # same learning for training and testing data set

    # this should work on all the columns autodetected as time or name given as time
    #list of already time identified columns
    ph = list(data.select_dtypes(include='datetime64').columns)
    # append it to the time feature argument
    self.time_feature = self.time_feature + ph
    # just make sure no colum appears double
    self.time_feature  = list(set(self.time_feature))
    # only do this if we have any date columns
    if len(self.time_feature) > 0:
      for t in self.time_feature:
        self.list_of_features = [t+"_"+i for i in self.list_of_features_o]
        data = add_datepart(data,t,prefix=t+"_")
        #  now we need to only keep thoes columns that user requested  so we need to do following procedure
        # we know following columns are generated by date part, save it as list
        cols = ['Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start']
        cols = [t+"_"+i for i in cols]
        
        for i in cols:
          if i not in self.list_of_features:
            data.drop(i,axis=1,inplace=True)

        # we want  time to be a catagorical variable 
        for i in self.list_of_features:
          data[i]= data[i].astype(str)
        # just get rid of Elapsed columns

        data.drop(t+"_"+"Elapsed",axis=1,inplace=True)
    
    return(data)
  
  def fit_transform(self,data,y=None):
    #self.fit(data)
    return(self.transform(data))

# _______________________________________________________________________________________________________________________
# make dummy variables
class Dummify(BaseEstimator,TransformerMixin):
  '''
    - makes one hot encoded variables for dummy variable
    - it is HIGHLY recommended to run the Select_Data_Type class first
    - Ignores target variable

      Args: 
        target: string , name of the target variable
        drop_first: boolen , default False. 
          -Wheather to get k-1 dummies out of K catagorical levels
  '''

  def __init__(self,target):
    self.target = target
    
    # creat ohe object 
    self.ohe = OneHotEncoder(handle_unknown='ignore')
  
  def fit(self,data,y=None):
    # will only do this if there are catagorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # we need to learn the column names once the training data set is dummify
      # save non catagorical data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      self.target_column =  data[[self.target]]
      # plus we will only take object data types
      self.data_columns  = pd.get_dummies(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object')),drop_first=self.drop_first).columns
      # now fit the trainin column
      self.ohe.fit(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object')))
    return(None)
 
  def transform(self,data,y=None):
    # will only do this if there are catagorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # only for test data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      # fit without target and only catagorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      #now put target , numerical and catagorical variables back togather
      data = pd.concat((self.data_nonc,data_dummies),axis=1)
      del(self.data_nonc)
      return(data)
    else:
      return(data)

  def fit_transform(self,data,y=None):
    # will only do this if there are catagorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      self.fit(data)
      # fit without target and only catagorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      # now put target , numerical and catagorical variables back togather
      data = pd.concat((self.target_column,self.data_nonc,data_dummies),axis=1)
      # remove unwanted attributes
      del(self.target_column,self.data_nonc)
      return(data)
    else:
      return(data)

#_________________________________________________________________________________________________________________________________________
class Fix_multicollinearity(BaseEstimator,TransformerMixin):
  """
          Fixes multicollinearity between predictor variables , also considering the correlation between target variable.
          Only applies to regression or two class classification ML use case
            Args:
              threshold (float): The utmost absolute pearson correlation tolerated beyween featres from 0.0 to 1.0
              target_variable (str): The target variable/column name
              correlation_with_target_threshold: minimum absolute correlation required between every feature and the target variable , default 1.0 (0.0 to 1.0)
              correlation_with_target_preference: float (0.0 to 1.0), default .08 ,while choosing between a pair of features w.r.t multicol & correlation target , this gives 
              the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
              a value of .5 means both measure have equal say 
  """
  # mamke a constructer
  
  def __init__ (self,threshold,target_variable,correlation_with_target_threshold= 0.0,correlation_with_target_preference=.80):
      self.threshold = threshold
      self.target_variable = target_variable
      self.correlation_with_target_threshold= correlation_with_target_threshold
      self.target_corr_weight = correlation_with_target_preference
      self.multicol_weight = 1-correlation_with_target_preference
  
  # Make fit method
  
  def fit (self,data,y=None):
    '''
        Args:
            data = takes preprocessed data frame
        Returns:
            None
    '''
      
    import numpy as np
    import numpy as np
    import pandas as pd
    #global data1
    self.data1 = data.copy()
    # make an correlation db with abs correlation db
    # self.data_c = self.data1.T.drop_duplicates()
    # self.data1 = self.data_c.T
    corr = pd.DataFrame(np.corrcoef(self.data1.T))
    corr.columns = self.data1.columns
    corr.index = self.data1.columns
    # self.corr_matrix = abs(self.data1.corr())
    self.corr_matrix = abs(corr)

    # for every diagonal value, make it Nan
    self.corr_matrix.values[tuple([np.arange(self.corr_matrix.shape[0])]*2)] = np.NaN
    
    # Now Calculate the average correlation of every feature with other, and get a pandas data frame
    self.avg_cor = pd.DataFrame(self.corr_matrix.mean())
    self.avg_cor['feature']= self.avg_cor.index
    self.avg_cor.reset_index(drop=True, inplace=True)
    self.avg_cor.columns =  ['avg_cor','features']
    
    # Calculate the correlation with the target
    self.targ_cor = pd.DataFrame(self.corr_matrix[self.target_variable].dropna())
    self.targ_cor['feature']= self.targ_cor.index
    self.targ_cor.reset_index(drop=True, inplace=True)
    self.targ_cor.columns =  ['target_variable','features']
    
    # Now, add a column for variable name and drop index
    self.corr_matrix['column'] = self.corr_matrix.index
    self.corr_matrix.reset_index(drop=True,inplace=True)
    
    # now we need to melt it , so that we can correlation pair wise , with two columns 
    self.cols =self.corr_matrix.column
    self.melt = self.corr_matrix.melt(id_vars= ['column'],value_vars=self.cols).sort_values(by='value',ascending=False).dropna()

    # now bring in the avg correlation for first of the pair
    self.merge = pd.merge(self.melt,self.avg_cor,left_on='column',right_on='features').drop('features',axis=1)

    # now bring in the avg correlation for second of the pair
    self.merge = pd.merge(self.merge,self.avg_cor,left_on='variable',right_on='features').drop('features',axis=1)

    # now bring in the target correlation for first of the pair
    self.merge = pd.merge(self.merge,self.targ_cor,left_on='column',right_on='features').drop('features',axis=1)

    # now bring in the avg correlation for second of the pair
    self.merge = pd.merge(self.merge,self.targ_cor,left_on='variable',right_on='features').drop('features',axis=1)

    # sort and save
    self.merge = self.merge.sort_values(by='value',ascending=False)

    # we need to now eleminate all the pairs that are actually duplicate e.g cor(x,y) = cor(y,x) , they are the same , we need to find these and drop them
    self.merge['all_columns'] = self.merge['column'] + self.merge['variable']

    # this puts all the coresponding pairs of features togather , so that we can only take one, since they are just the duplicates
    self.merge['all_columns'] = [sorted(i) for i in self.merge['all_columns'] ]

    # now sort by new column
    self.merge = self.merge.sort_values(by='all_columns')

    # take every second colums
    self.merge = self.merge.iloc[::2, :]

    # make a ranking column to eliminate features
    self.merge['rank_x'] = round(self.multicol_weight*(self.merge['avg_cor_y']- self.merge['avg_cor_x']) +  self.target_corr_weight*(self.merge['target_variable_x'] - self.merge['target_variable_y']),6) # round it to 6 digits
    self.merge1 = self.merge # delete here
    ## Now there will be rows where the rank will be exactly zero, these is where the value (corelartion between features) is exactly one ( like price and price^2)
    ## so in that case , we can simply pick one of the variable
    # but since , features can be in either column, we will drop one column (say 'column') , only if the feature is not in the second column (in variable column)
    # both equations below will return the list of columns to drop from here 
    # this is how it goes

    ## For the portion where correlation is exactly one !
    self.one = self.merge[self.merge['rank_x']==0]

    # this portion is complicated 
    # table one have all the paired variable having corelation of 1
    # in a nutshell, we can take any column (one side of pair) and delete the other columns (other side of the pair)
    # however one varibale can appear more than once on any of the sides , so we will run for loop to find all pairs...
    # here it goes
    # take a list of all (but unique ) variables that have correlation 1 for eachother, we will make two copies
    self.u_all = list(pd.unique(pd.concat((self.one['column'],self.one['variable']),axis=0)))
    self.u_all_1 = list(pd.unique(pd.concat((self.one['column'],self.one['variable']),axis=0)))
    # take a list of features (unique) for the first side of the pair
    self.u_column  = pd.unique(self.one['column'])
    
    # now we are going to start picking each variable from one column (one side of the pair) , check it against the other column (other side of the pair)
    # to pull all coresponding / paired variables  , and delete thoes newly varibale names from all unique list
    
    for i in self.u_column:
      #print(i)
      r = self.one[self.one['column']==i]['variable'].values
      for q in r:
        if q in self.u_all:
          #print("_"+q)
          self.u_all.remove(q)

    # now the unique column contains the varibales that should remain, so in order to get the variables that should be deleted :
    self.to_drop =(list(set(self.u_all_1)-set(self.u_all)))


    # self.to_drop_a =(list(set(self.one['column'])-set(self.one['variable'])))
    # self.to_drop_b =(list(set(self.one['variable'])-set(self.one['column'])))
    # self.to_drop = self.to_drop_a + self.to_drop_b

    ## now we are to treat where rank is not Zero and Value (correlation) is greater than a specific threshold
    self.non_zero = self.merge[(self.merge['rank_x']!= 0.0) & (self.merge['value'] >= self.threshold)]

    # pick the column to delete
    self.non_zero_list = list(np.where(self.non_zero['rank_x'] < 0 , self.non_zero['column'], self.non_zero['variable']))

    # add two list
    self.to_drop = self.to_drop + self.non_zero_list

    #make sure that target column is not a part of the list
    try:
        self.to_drop.remove(self.target_variable)
    except:
        self.to_drop
    
    self.to_drop = self.to_drop

    # now we want to keep only the columns that have more correlation with traget by a threshold
    self.to_drop_taret_correlation=[] 
    if self.correlation_with_target_threshold != 0.0:
      corr = pd.DataFrame(np.corrcoef(data.drop(self.to_drop,axis=1).T),columns= data.drop(self.to_drop,axis=1).columns,index=data.drop(self.to_drop,axis=1).columns)
      self.to_drop_taret_correlation = corr[self.target_variable].abs()
      # self.to_drop_taret_correlation = data.drop(self.to_drop,axis=1).corr()[self.target_variable].abs()
      self.to_drop_taret_correlation = self.to_drop_taret_correlation [self.to_drop_taret_correlation < self.correlation_with_target_threshold ]
      self.to_drop_taret_correlation = list(self.to_drop_taret_correlation.index)
      #self.to_drop = self.corr + self.to_drop
      try:
        self.to_drop_taret_correlation.remove(self.target_variable)
      except:
        self.to_drop_taret_correlation
      

  # now Transform
  def transform(self,data,y=None):
    '''
        Args:f
            data = takes preprocessed data frame
        Returns:
            data frame
    '''
    data.drop(self.to_drop,axis=1,inplace=True)
    # now drop less correlated data
    data.drop(self.to_drop_taret_correlation,axis=1,inplace=True,errors='ignore')
    return(data)
  
  # fit_transform
  def fit_transform(self,data, y=None):
    
    '''
        Args:
            data = takes preprocessed data frame
        Returns:
            data frame
    '''
    
    self.fit(data)
    return(self.transform(data))

#____________________________________________________________________________________________________________________________________________________________________
# Empty transformer
class Empty(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return same DF 
  '''

  def __init__(self):
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,data,y=None):
    return(data)

  def fit_transform(self,data,y=None):
    return(self.transform(data))

# ______________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one
def Preprocess_Path_One(train_data,target_variable,ml_usecase=None,test_data =None,catagorical_features=[],numerical_features=[],time_features=[],time_features_extracted=['Month','Dayofweek'],
                               imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',catagorical_imputation_strategy='not_available'
                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - 1) Auto infer data types
      - 2) Impute (simple or with surrogate columns)
      - 3) One Hot / Dummy encoding
  '''

  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  if ml_usecase is None:
    if ( ( (c1) & (c2) ) | (c3)   ):
      ml_usecase ='classification'
    else:
      ml_usecase ='regression'
  
  
  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,catagorical_features=catagorical_features,numerical_features=numerical_features,time_features=time_features)
  # dtypes1 = Dtypes2()
  
  # for imputation
  if imputation_type == "simple imputer":
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,catagorical_strategy=catagorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,catagorical_strategy=catagorical_imputation_strategy,target_variable=target_variable)
  
  # for Time Variables
  feature_time = Make_Time_Features(time_feature=time_features,list_of_features=time_features_extracted)
  dummy = Dummify(target_variable)
  

  fixm = Fix_multicollinearity(threshold=1.0,target_variable=target_variable,correlation_with_target_threshold=0.0) # this will work even for multi classidifcation, becase we are only trying to eliminate multicol of exactly 1


  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('feature_time',feature_time),
                 ('dummy',dummy),
                 ('fixm',fixm)
                 ])
  
  if test_data is not None:
    return(pipe.fit_transform(train_data),pipe.transform(test_data))
  else:
    return(pipe.fit_transform(train_data))

