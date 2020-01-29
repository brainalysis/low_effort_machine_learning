import pandas as pd
import numpy as np
import ipywidgets as wg 
from IPython.display import display
from ipywidgets import Layout
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import KBinsDiscretizer
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA as PCA_od
from sklearn import cluster
import sys 
from sklearn.pipeline import Pipeline
from sklearn import metrics
import datefinder
from datetime import datetime
import calendar
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#ignore warnings
import warnings
warnings.filterwarnings('ignore') 

#_____________________________________________________________________________________________________________________________

class DataTypes_Auto_infer(BaseEstimator,TransformerMixin):
  '''
    - This will try to infer data types automatically, option to override learent data types is also available.
    - This alos automatically delets duplicate columns (values or same colume name), removes rows where target variable is null and 
      remove columns and rows where all the records are null
  '''

  def __init__(self,target,ml_usecase,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=True): # nothing to define
    '''
    User to define the target (y) variable
      args:
        target: string, name of the target variable
        ml_usecase: string , 'regresson' or 'classification . For now, only supports two  class classification
        - this is useful in case target variable is an object / string . it will replace the strings with integers
        categorical_features: list of categorical features, default None, when None best guess will be used to identify categorical features
        numerical_features: list of numerical features, default None, when None best guess will be used to identify numerical features
        time_features: list of date/time features, default None, when None best guess will be used to identify date/time features    
  '''
    self.target = target
    self.ml_usecase= ml_usecase
    self.categorical_features =categorical_features
    self.numerical_features = numerical_features
    self.time_features =time_features
    self.features_todrop = features_todrop
    self.display_types = display_types
  
  def fit(self,dataset,y=None): # learning data types of all the columns
    '''
    Args: 
      data: accepts a pandas data frame
    Returns:
      Panda Data Frame
    '''
    data = dataset.copy()
    # remove sepcial char from column names
    #data.columns= data.columns.str.replace('[,]','')

    # we will take float as numberic, object as categorical from the begning
    # fir int64, we will check to see what is the proportion of unique counts to the total lenght of the data
    # if proportion is lower, then it is probabaly categorical 
    # however, proportion can be lower / disturebed due to samller denominator (total lenghth / number of samples)
    # so we will take the following chart
    # 0-50 samples, threshold is 24%
    # 50-100 samples, th is 12%
    # 50-250 samples , th is 4.8%
    # 250-500 samples, th is 2.4%
    # 500 and above 2% or belwo
   
    # if there are inf or -inf then replace them with NaN
    data.replace([np.inf,-np.inf],np.NaN,inplace=True)
   
    # we canc check if somehow everything is object, we can try converting them in float
    for i in data.select_dtypes(include=['object']).columns:
      try:
        data[i] = data[i].astype('int64')
      except:
        None
    
    # if data type is bool , convert to categorical
    for i in data.columns:
      if data[i].dtype=='bool':
        data[i] = data[i].astype('object')
    

    # some times we have id column in the data set, we will try to find it and then  will drop it if found
    len_samples = len(data)
    self.id_columns = []
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtype in ['int64','float64']:
        if sum(data[i].isna()) == 0: 
          if len(data[i].unique()) == len_samples:
            min_number = min(data[i])
            max_number = max(data[i])
            arr = np.arange(min_number,max_number+1,1)
            try:
              all_match = sum(data[i].sort_values() == arr)
              if all_match == len_samples:
                self.id_columns.append(i) 
            except:
              None 
    
    data_len = len(data)                        
        
    # wiith csv , if we have any null in  a colum that was int , panda will read it as float.
    # so first we need to convert any such floats that have NaN and unique values are lower than 20
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtypes == 'float64':
        # count how many Nas are there
        na_count = sum(data[i].isna())
        # count how many digits are there that have decimiles
        count_float = np.nansum([ False if r.is_integer() else True for r in data[i]])
        # total decimiels digits
        count_float = count_float - na_count # reducing it because we know NaN is counted as a float digit
        # now if there isnt any float digit , & unique levales are less than 20 and there are Na's then convert it to object
        if ( (count_float == 0) & (len(data[i].unique()) <=20) & (na_count>0) ):
          data[i] = data[i].astype('object')
        

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
        if len(data[i].unique()) <=20: #hard coded
          data[i]= data[i].apply(str)
        else:
          data[i]= data[i].astype('float64')


    # # if colum is objfloat  and only have two unique counts , this is probabaly one hot encoded
    # # make it object
    for i in data.columns:
      if ((data[i].dtypes == 'float64') & (len(data[i].unique())==2)):
        data[i]= data[i].apply(str)
    
    
    #for time & dates
    #self.drop_time = [] # for now we are deleting time columns
    for i in data.drop(self.target,axis=1).columns:
      # we are going to check every first row of every column and see if it is a date
      match = datefinder.find_dates(data[i].values[0]) # to get the first value
      try:
        for m in match:
          if isinstance(m, datetime) == True:
            data[i] = pd.to_datetime(data[i])
            #self.drop_time.append(i)  # for now we are deleting time columns
      except:
        continue

    # now in case we were given any specific columns dtypes in advance , we will over ride theos 
    if len(self.categorical_features) > 0:
      for i in self.categorical_features:
        data[i]=data[i].apply(str)
    
    if len(self.numerical_features) > 0:
      for i in self.numerical_features:
        data[i]=data[i].astype('float64')
    
    if len(self.time_features) > 0:
      for i in self.time_features:
        data[i]=pd.to_datetime(data[i])

    # table of learent types
    self.learent_dtypes = data.dtypes
    #self.training_columns = data.drop(self.target,axis=1).columns

    # lets remove duplicates
    # remove duplicate columns (columns with same values)
    #(too expensive on bigger data sets)
    # data_c = data.T.drop_duplicates()
    # data = data_c.T
    #remove columns with duplicate name 
    data = data.loc[:,~data.columns.duplicated()]
    # Remove NAs
    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    # remove the row if target column has NA
    data = data[~data[self.target].isnull()]
            

    #self.training_columns = data.drop(self.target,axis=1).columns

    # since due to transpose , all data types have changed, lets change the dtypes to original---- not required any more since not transposing any more
    # for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
    #   data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    
    if self.display_types == True:
      display(wg.Text(value="Following data types have been inferred automatically, if they are correct press enter to continue or type 'quit' otherwise.",layout =Layout(width='100%')))
      
      dt_print_out = pd.DataFrame(self.learent_dtypes, columns=['Feature_Type'])
      dt_print_out['Data Type'] = ""
      
      for i in dt_print_out.index:
        if i != self.target:
          if dt_print_out.loc[i,'Feature_Type'] == 'object':
            dt_print_out.loc[i,'Data Type'] = 'Categorical'
          elif dt_print_out.loc[i,'Feature_Type'] == 'float64':
            dt_print_out.loc[i,'Data Type'] = 'Numeric'
          elif dt_print_out.loc[i,'Feature_Type'] == 'datetime64[ns]':
            dt_print_out.loc[i,'Data Type'] = 'Date'
          #elif dt_print_out.loc[i,'Feature_Type'] == 'int64':
          #  dt_print_out.loc[i,'Data Type'] = 'Categorical'
        else:
          dt_print_out.loc[i,'Data Type'] = 'Label'

      # for ID column:
      for i in dt_print_out.index:
        if i in self.id_columns:
          dt_print_out.loc[i,'Data Type'] = 'ID Column'
      
      # if we added the dummy  target column , then drop it 
      dt_print_out.drop(index='dummy_target',errors='ignore',inplace=True)
      # drop any columns that were asked to drop
      dt_print_out.drop(index=self.features_todrop,errors='ignore',inplace=True)


      display(dt_print_out[['Data Type']])
      self.response = input()

      if self.response in ['quit','Quit','exit','EXIT','q','Q','e','E','QUIT','Exit']:
        sys.exit('Read the documentation of setup to learn how to overwrite data types over the inferred types. setup function must run again before you continue modeling.')
    
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)
    
    return(data)
  
  def transform(self,dataset,y=None):
    '''
      Args: 
        data: accepts a pandas data frame
      Returns:
        Panda Data Frame
    '''
    data = dataset.copy()
    # remove sepcial char from column names
    #data.columns= data.columns.str.replace('[,]','')

    #very first thing we need to so is to check if the training and test data hace same columns
    #exception checking   
    import sys

    for i in self.final_training_columns:  
      if i not in data.columns:
        sys.exit('(Type Error): test data does not have column ' + str(i) + " which was used for training")

    ## we only need to take test columns that we used in ttaining (test in production may have a lot more columns)
    data = data[self.final_training_columns]

    
    # just keep picking the data and keep applying to the test data set (be mindful of target variable)
    for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
      data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)

     # drop custome columns
    data.drop(self.features_todrop,axis=1,errors='ignore',inplace=True)
    
    return(data)

  # fit_transform
  def fit_transform(self,dataset,y=None):

    data= dataset.copy()
    # since this is for training , we dont nees any transformation since it has already been transformed in fit
    data = self.fit(data)

    # additionally we just need to treat the target variable
    # for ml use ase
    if ((self.ml_usecase == 'classification') &  (data[self.target].dtype=='object')):
      le = LabelEncoder()
      data[self.target] = le.fit_transform(np.array(data[self.target]))

      # now get the replacement dict
      rev= le.inverse_transform(range(0,len(le.classes_)))
      rep = np.array(range(0,len(le.classes_)))
      self.replacement={}
      for i,k in zip(rev,rep):
        self.replacement[i] = k

      # self.u = list(pd.unique(data[self.target]))
      # self.replacement = np.arange(0,len(self.u))
      # data[self.target]= data[self.target].replace(self.u,self.replacement)
      # data[self.target] = data[self.target].astype('int64')
      # self.replacement = pd.DataFrame(dict(target_variable=self.u,replaced_with=self.replacement))

    
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)

    # drop custome columns
    data.drop(self.features_todrop,axis=1,errors='ignore',inplace=True)
    
    # finally save a list of columns that we would need from test data set
    self.final_training_columns = data.drop(self.target,axis=1).columns

    
    return(data)
# _______________________________________________________________________________________________________________________
# Imputation
class Simple_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes all type of data (numerical,categorical & Time).
      Highly recommended to run Define_dataTypes class first
      Numerical values can be imputed with mean or median 
      categorical missing values will be replaced with "Other"
      Time values are imputed with the most frequesnt value
      Ignores target (y) variable    
      Args: 
        Numeric_strategy: string , all possible values {'mean','median'}
        categorical_strategy: string , all possible values {'not_available','most frequent'}
        target: string , name of the target variable

  '''

  def __init__(self,numeric_strategy,categorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.categorical_strategy = categorical_strategy
  
  def fit(self,dataset,y=None): #
    data = dataset.copy()
    # make a table for numerical variable with strategy stats
    if self.numeric_strategy == 'mean':
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmean)
    else:
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmedian)

    self.numeric_columns = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).columns

    #for Catgorical , 
    if self.categorical_strategy == 'most frequent':
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_stats = pd.DataFrame(columns=self.categorical_columns) # place holder
      for i in (self.categorical_stats.columns):
        self.categorical_stats.loc[0,i] = data[i].value_counts().index[0]
    else:
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
    
    # for time, there is only one way, pick up the most frequent one
    self.time_columns = data.drop(self.target,axis=1).select_dtypes(include=['datetime64[ns]']).columns
    self.time_stats = pd.DataFrame(columns=self.time_columns) # place holder
    for i in (self.time_columns):
      self.time_stats.loc[0,i] = data[i].value_counts().index[0]
    return(data)

      
  
  def transform(self,dataset,y=None):
    data = dataset.copy() 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      data[i].fillna(s,inplace=True)
    
    # for categorical columns
    if self.categorical_strategy == 'most frequent':
      for i in (self.categorical_stats.columns):
        #data[i].fillna(self.categorical_stats.loc[0,i],inplace=True)
        data[i] = data[i].fillna(self.categorical_stats.loc[0,i])
        data[i] = data[i].apply(str)    
    else: # this means replace na with "not_available"
      for i in (self.categorical_columns):
        data[i].fillna("not_available",inplace=True)
        data[i] = data[i].apply(str)
    # for time
    for i in (self.time_stats.columns):
        data[i].fillna(self.time_stats.loc[0,i],inplace=True)
    
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy() 
    data= self.fit(data)
    return(self.transform(data))

# _______________________________________________________________________________________________________________________
# Imputation with surrogate columns
class Surrogate_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes feature with surrogate column (numerical,categorical & Time).
      - Highly recommended to run Define_dataTypes class first
      - it is also recommended to only apply this to features where it makes business sense to creat surrogate column
      - feature name has to be provided
      - only able to handle one feature at a time
      - Numerical values can be imputed with mean or median 
      - categorical missing values will be replaced with "Other"
      - Time values are imputed with the most frequesnt value
      - Ignores target (y) variable    
      Args: 
        feature_name: string, provide features name
        feature_type: string , all possible values {'numeric','categorical','date'}
        strategy: string ,all possible values {'mean','median','not_available','most frequent'}
        target: string , name of the target variable

  '''
  def __init__(self,numeric_strategy,categorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.categorical_strategy = categorical_strategy
  
  def fit(self,dataset,y=None): #
    data = dataset.copy()
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
    if self.categorical_strategy == 'most frequent':
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_stats = pd.DataFrame(columns=self.categorical_columns) # place holder
      for i in (self.categorical_stats.columns):
        self.categorical_stats.loc[0,i] = data[i].value_counts().index[0]
      # also need to learn if any columns had NA in training, but this is only valid if strategy is "most frequent"
      self.categorical_na = pd.DataFrame(columns=self.categorical_columns)
      for i in self.categorical_columns:
        if sum(data[i].isna()) > 0:
          self.categorical_na.loc[0,i] = True
        else:
          self.categorical_na.loc[0,i] = False        
    else:
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_na = pd.DataFrame(columns=self.categorical_columns)
      self.categorical_na.loc[0,:] = False #(in this situation we are not making any surrogate column)
    
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

      
  
  def transform(self,dataset,y=None):
    data = dataset.copy() 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      array = data[i].isna()
      data[i].fillna(s,inplace=True)
      # make a surrogate column if there was any
      if self.numeric_na.loc[0,i] == True:
        data[i+"_surrogate"]= array
        # make it string
        data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)

    
    # for categorical columns
    if self.categorical_strategy == 'most frequent':
      for i in (self.categorical_stats.columns):
        #data[i].fillna(self.categorical_stats.loc[0,i],inplace=True)
        array = data[i].isna()
        data[i] = data[i].fillna(self.categorical_stats.loc[0,i])
        data[i] = data[i].apply(str)  
        # make surrogate column
        if self.categorical_na.loc[0,i] == True:
          data[i+"_surrogate"]= array
          # make it string
          data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)
    else: # this means replace na with "not_available"
      for i in (self.categorical_columns):
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
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    data= self.fit(data)
    return(self.transform(data))
# _______________________________________________________________________________________________________________________
# Zero and Near Zero Variance
class Zroe_NearZero_Variance(BaseEstimator,TransformerMixin):
  '''
    - it eliminates the features having zero variance
    - it eliminates the features haveing near zero variance
    - Near zero variance is determined by 
      -1) Count of unique points divided by the total length of the feature has to be lower than a pre sepcified threshold 
      -2) Most common point(count) divided by the second most common point(count) in the feature is greater than a pre specified threshold
      Once both conditions are met , the feature is dropped  
    -Ignores target variable
      
      Args: 
        threshold_1: float (between 0.0 to 1.0) , default is .10 
        threshold_2: int (between 1 to 100), default is 20 
        tatget variable : string, name of the target variable

  '''

  def __init__(self,target,threshold_1=0.1,threshold_2=20):
    self.threshold_1 = threshold_1
    self.threshold_2 = threshold_2
    self.target = target
  
  def fit(self,dataset,y=None): # from training data set we are going to learn what columns to drop
    data = dataset.copy()
    self.to_drop= []
    self.sampl_len = len(data[self.target])
    for i in data.drop(self.target,axis=1).columns:
      # get the number of unique counts
      u = pd.DataFrame( data[i].value_counts()).sort_values(by=i,ascending=False, inplace=False)
      # take len of u and divided it by the total sample numbers, so this will check the 1st rule , has to be low say 10%
      #import pdb; pdb.set_trace()
      first=len(u)/self.sampl_len
      # then check if most common divided by 2nd most common ratio is 20 or more
      if len(u[i]) == 1: # this means that if column is non variance , automatically make the number big to drop it
        second=100
      else:
        second = u.iloc[0,0]/u.iloc[1,0]
    # if both conditions are true then drop the column, however, we dont want to alter column that indicate NA's
      if ((first <= 0.10) and (second >=20) and (i[-10:]!='_surrogate')):
        self.to_drop.append(i) 
    # now drop if the column has zero variance
      if (((second ==100) and (i[-10:]!='_surrogate'))):
        self.to_drop.append(i) 

  
  def transform(self,dataset,y=None): # since it is only for training data set , nothing here
    data= dataset.copy()
    data.drop(self.to_drop,axis=1,inplace=True)
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data= dataset.copy()
    self.fit(data)
    return(self.transform(data))
#____________________________________________________________________________________________________________________________
# rare catagorical variables
class Catagorical_variables_With_Rare_levels(BaseEstimator,TransformerMixin):
  '''
    -Merges levels in catagorical features with more frequent level  if they appear less than a threshold count 
      e.g. Col=[a,a,a,a,b,b,c,c]
      if threshold is set to 2 , then c will be mrged with b because both are below threshold
      There has to be atleast two levels belwo threshold for this to work 
      the process will keep going until all the levels have atleast 2(threshold) counts
    -Only handles catagorical features
    -It is recommended to run the Zroe_NearZero_Variance and Define_dataTypes first
    -Ignores target variable 
      Args: 
        threshold: int , default 10
        target: string , name of the target variable
        new_level_name: string , name given to the new level generated, default 'others'

  '''

  def __init__(self,target,new_level_name='others_infrequent',threshold=.05):
    self.threshold = threshold
    self.target = target
    self.new_level_name = new_level_name
  def fit(self,dataset,y=None): # we will learn for what columnns what are the level to merge as others
    # every level of the catagorical feature has to be more than threshols, if not they will be clubed togather as "others"
    # in order to apply, there should be atleast two levels belwo the threshold ! 
    # creat a place holder
    data = dataset.copy()
    self.ph = pd.DataFrame(columns=data.drop(self.target,axis=1).select_dtypes(include="object").columns)
    #ph.columns = df.columns# catagorical only 
    for i in data[self.ph.columns].columns:
        # determine the infrequebt count
        v_c = data[i].value_counts()
        count_th = round(v_c.quantile(self.threshold))
        a = np.sum(pd.DataFrame(data[i].value_counts().sort_values()) [i]  <= count_th)
        if a >= 2: # rare levels has to be atleast two
          count = pd.DataFrame( data[i].value_counts().sort_values())
          count.columns = ['fre']
          count = count[count['fre']<=count_th]
          to_club = list(count.index)
          self.ph.loc[0,i] = to_club
        else:
          self.ph.loc[0,i] = []
    # # also need to make a place holder that keep records of all the levels , and in case a new level appears in test we will change it to others
    # self.ph_level = pd.DataFrame(columns=data.drop(self.target,axis=1).select_dtypes(include="object").columns)
    # for i in self.ph_level.columns:
    #   self.ph_level.loc[0,i] = list(data[i].value_counts().sort_values().index)

  
  def transform(self,dataset,y=None): # 
    # transorm 
    data = dataset.copy()
    for i in data[self.ph.columns].columns:
      t_replace = self.ph.loc[0,i]
      data[i].replace(to_replace=t_replace,value=self.new_level_name,inplace=True)
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    self.fit(data)
    return(self.transform(data))


#____________________________________________________________________________________________________________________________________________________________________
# Binning for Continious
class Binning(BaseEstimator,TransformerMixin):
  '''
    - Converts numerical variables to catagorical variable through binning
    - Number of binns are automitically determined through Sturges method
    - Once discretize, original feature will be dropped
        Args:
            features_to_discretize: list of featur names to be binned

  '''

  def __init__(self, features_to_discretize):
    self.features_to_discretize =features_to_discretize
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()
    #only do if features are provided
    if len(self.features_to_discretize) > 0:
      data_t = self.disc.transform(np.array(data[self.features_to_discretize]).reshape(-1,self.len_columns))
      # make pandas data frame
      data_t = pd.DataFrame(data_t,columns=self.features_to_discretize,index=data.index)
      # all these columns are catagorical
      data_t = data_t.astype(str)
      # drop original columns
      data.drop(self.features_to_discretize,axis=1,inplace=True)
      # add newly created columns
      data = pd.concat((data,data_t),axis=1)
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # only do if features are given

    if len(self.features_to_discretize) > 0:

      # place holder for all the features for their binns  
      self.binns = []
      for i in self.features_to_discretize:
        # get numbr of binns
        hist, bin_edg = np.histogram(data[i],bins='sturges')
        self.binns.append(len(hist))

      # how many colums to deal with
      self.len_columns = len(self.features_to_discretize)
      # now do fit transform 
      self.disc = KBinsDiscretizer(n_bins=self.binns, encode='ordinal', strategy='kmeans')
      data_t = self.disc.fit_transform(np.array(data[self.features_to_discretize]).reshape(-1,self.len_columns))
      # make pandas data frame
      data_t = pd.DataFrame(data_t,columns=self.features_to_discretize,index=data.index)
      # all these columns are catagorical
      data_t = data_t.astype(str)
      # drop original columns
      data.drop(self.features_to_discretize,axis=1,inplace=True)
      # add newly created columns
      data = pd.concat((data,data_t),axis=1)

    return(data)
# ______________________________________________________________________________________________________________________
# Scaling & Power Transform
class Scaling_and_Power_transformation(BaseEstimator,TransformerMixin):
  '''
    -Given a data set, applies Min Max, Standar Scaler or Power Transformation (yeo-johnson)
    -it is recommended to run Define_dataTypes first
    - ignores target variable 
      Args: 
        target: string , name of the target variable
        function_to_apply: string , default 'zscore' (standard scaler), all other {'minmaxm','yj','quantile','robust','maxabs'} ( min max,yeo-johnson & quantile power transformation, robust and MaxAbs scaler )

  '''

  def __init__(self,target,function_to_apply='zscore',random_state_quantile=42):
    self.target = target
    self.function_to_apply = function_to_apply
    self.random_state_quantile = random_state_quantile
    # self.transform_target = transform_target
    # self.ml_usecase = ml_usecase
  
  def fit(self,dataset,y=None):
    
    data=dataset.copy()
    # we only want to apply if there are numeric columns
    self.numeric_features = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=["float64",'int64']).columns
    if len(self.numeric_features) > 0:
      if self.function_to_apply == 'zscore':
        self.scale_and_power = StandardScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'minmax':
        self.scale_and_power = MinMaxScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'yj':
        self.scale_and_power = PowerTransformer(method='yeo-johnson',standardize=False)
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'quantile':
        self.scale_and_power = QuantileTransformer(random_state=self.random_state_quantile,output_distribution='normal')
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'robust':
        self.scale_and_power = RobustScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'maxabs':
        self.scale_and_power = MaxAbsScaler()
        self.scale_and_power.fit(data[self.numeric_features])

      else:
        return(None)
    else:
      return(None)
    

  
  def transform(self,dataset,y=None):
    data = dataset.copy()
    
    if len(self.numeric_features) > 0:
      self.data_t = pd.DataFrame(self.scale_and_power.transform(data[self.numeric_features]))
      # we need to set the same index as original data
      self.data_t.index = data.index
      self.data_t.columns = self.numeric_features
      for i in self.numeric_features:
        data[i]= self.data_t[i]
      return(data)
    
    else:
      return(data) 

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    self.fit(data)
    # convert target if appropriate
    # default behavious is quantile transformer
    # if ((self.ml_usecase == 'regression') and (self.transform_target == True)):
    #   self.scale_and_power_target = QuantileTransformer(random_state=self.random_state_quantile,output_distribution='normal')
    #   data[self.target]=self.scale_and_power_target.fit_transform(np.array(data[self.target]).reshape(-1,1))
      
    return(self.transform(data))

# ______________________________________________________________________________________________________________________
# Scaling & Power Transform
class Target_Transformation(BaseEstimator,TransformerMixin):
  '''
    - Applies Power Transformation (yeo-johnson , Box-Cox) to target variable (Applicable to Regression only)
      - 'bc' for Box_Coc & 'yj' for yeo-johnson, default is Box-Cox
    - if target containes negtive / zero values , yeo-johnson is automatically selected 
    
  '''

  def __init__(self,target,function_to_apply='bc'):
    self.target = target
    self.function_to_apply = function_to_apply
    if self.function_to_apply == 'bc':
      self.function_to_apply = 'box-cox'
    else:
      self.function_to_apply = 'yeo-johnson'

  
  def fit(self,dataset,y=None):
    return(None)
    

  
  def transform(self,dataset,y=None):
    return(dataset) 

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # if target has zero or negative values use yj instead
    if any(data[self.target]<= 0):
      self.function_to_apply = 'yeo-johnson'
    # apply transformation
    self.p_transform_target = PowerTransformer(method=self.function_to_apply)
    data[self.target]=self.p_transform_target.fit_transform(np.array(data[self.target]).reshape(-1,1))
      
    return(data)
# __________________________________________________________________________________________________________________________
# Time feature extractor
class Make_Time_Features(BaseEstimator,TransformerMixin):
  '''
    -Given a time feature , it extracts more features
    - Only accepts / works where feature / data type is datetime64[ns]
    - full list of features is:
      ['month','weekday',is_month_end','is_month_start','hour']
    - all extracted features are defined as string / object
    -it is recommended to run Define_dataTypes first
      Args: 
        time_feature: list of feature names as datetime64[ns] , default empty/none , if empty/None , it will try to pickup dates automatically where data type is datetime64[ns]
        list_of_features: list of required features , default value ['month','weekday','is_month_end','is_month_start','hour']

  '''

  def __init__(self,time_feature=[],list_of_features=['month','weekday','is_month_end','is_month_start','hour']):
    self.time_feature = time_feature
    self.list_of_features_o = list_of_features
    return(None)

  def fit(self,data,y=None):

    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()

    # run fit transform first

    # start making features for every column in the time list
    for i in self.time_feature:
      # make month column if month is choosen
      if 'month' in self.list_of_features_o:
        data[i+"_month"] = [datetime.date(r).month for r in data[i]]
        data[i+"_month"] = data[i+"_month"].apply(str)

      # make weekday column if weekday is choosen ( 0 for monday 6 for sunday)
      if 'weekday' in self.list_of_features_o:
        data[i+"_weekday"] = [datetime.weekday(r) for r in data[i]]
        data[i+"_weekday"] = data[i+"_weekday"].apply(str)
      
      # make Is_month_end column  choosen
      if 'is_month_end' in self.list_of_features_o:
        data[i+"_is_month_end"] = [ 1 if calendar.monthrange(datetime.date(r).year,datetime.date(r).month)[1] == datetime.date(r).day  else 0 for r in data[i] ]
        data[i+"_is_month_end"] = data[i+"_is_month_end"].apply(str)
        
      
      # make Is_month_start column if choosen
      if 'is_month_start' in self.list_of_features_o:
        data[i+"_is_month_start"] = [ 1 if datetime.date(r).day == 1 else 0 for r in data[i] ]
        data[i+"_is_month_start"] = data[i+"_is_month_start"].apply(str)
      
      # make hour column if choosen
      if 'hour' in self.list_of_features_o:
        h = [ datetime.time(r).hour for r in data[i] ]
        if sum(h) > 0:  
          data[i+"_hour"] = h
          data[i+"_hour"] = data[i+"_hour"].apply(str)
    
    # we dont need time columns any more 
    data.drop(self.time_feature,axis=1,inplace=True)

    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # if no columns names are given , then pick datetime columns
    if len(self.time_feature) == 0 :
      self.time_feature = [i for i in data.columns if data[i].dtype == 'datetime64[ns]']
    
    # now start making features for every column in the time list
    for i in self.time_feature:
      # make month column if month is choosen
      if 'month' in self.list_of_features_o:
        data[i+"_month"] = [datetime.date(r).month for r in data[i]]
        data[i+"_month"] = data[i+"_month"].apply(str)

      # make weekday column if weekday is choosen ( 0 for monday 6 for sunday)
      if 'weekday' in self.list_of_features_o:
        data[i+"_weekday"] = [datetime.weekday(r) for r in data[i]]
        data[i+"_weekday"] = data[i+"_weekday"].apply(str)
      
      # make Is_month_end column  choosen
      if 'is_month_end' in self.list_of_features_o:
        data[i+"_is_month_end"] = [ 1 if calendar.monthrange(datetime.date(r).year,datetime.date(r).month)[1] == datetime.date(r).day  else 0 for r in data[i] ]
        data[i+"_is_month_end"] = data[i+"_is_month_end"].apply(str)
        
      
      # make Is_month_start column if choosen
      if 'is_month_start' in self.list_of_features_o:
        data[i+"_is_month_start"] = [ 1 if datetime.date(r).day == 1 else 0 for r in data[i] ]
        data[i+"_is_month_start"] = data[i+"_is_month_start"].apply(str)
      
      # make hour column if choosen
      if 'hour' in self.list_of_features_o:
        h = [ datetime.time(r).hour for r in data[i] ]
        if sum(h) > 0:  
          data[i+"_hour"] = h
          data[i+"_hour"] = data[i+"_hour"].apply(str)
    
    # we dont need time columns any more 
    data.drop(self.time_feature,axis=1,inplace=True)

    return(data)


# _______________________________________________________________________________________________________________________

# make dummy variables
class Dummify(BaseEstimator,TransformerMixin):
  '''
    - makes one hot encoded variables for dummy variable
    - it is HIGHLY recommended to run the Select_Data_Type class first
    - Ignores target variable

      Args: 
        target: string , name of the target variable
  '''

  def __init__(self,target):
    self.target = target
    
    # creat ohe object 
    self.ohe = OneHotEncoder(handle_unknown='ignore')
  
  def fit(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # we need to learn the column names once the training data set is dummify
      # save non categorical data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      self.target_column =  data[[self.target]]
      # # plus we will only take object data types
      try:
        self.data_columns  = pd.get_dummies(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).columns
      except:
        self.data_columns = []
      # # now fit the trainin column
      self.ohe.fit(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object')))
    else:
      None
    return(None)
 
  def transform(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # only for test data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      # fit without target and only categorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      #now put target , numerical and categorical variables back togather
      data = pd.concat((self.data_nonc,data_dummies),axis=1)
      del(self.data_nonc)
      return(data)
    else:
      return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      self.fit(data)
      # fit without target and only categorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      # now put target , numerical and categorical variables back togather
      data = pd.concat((self.target_column,self.data_nonc,data_dummies),axis=1)
      # remove unwanted attributes
      del(self.target_column,self.data_nonc)
      return(data)
    else:
      return(data)

# _______________________________________________________________________________________________________________________
# Outlier
class Outlier(BaseEstimator,TransformerMixin):
  '''
    - Removes outlier using ABOD,KNN,IFO,PCA & HOBS using hard voting
    - Only takes numerical / One Hot Encoded features
  '''

  def __init__(self,target,contamination=.20, random_state=42):
    self.target = target
    self.contamination = contamination
    self.random_state = random_state
    self.knn = KNN(contamination=self.contamination)
    self.iso = IForest(contamination=self.contamination,random_state=self.random_state)
    self.pca = PCA_od(contamination=self.contamination,random_state=self.random_state)

    return(None)

  def fit(self,data,y=None):
    #self.abod.fit(data)
    return(None)

  def transform(self,data,y=None):
    return(data)

  def fit_transform(self,dataset,y=None):

    # dummify if there are any obects 
    if len(dataset.select_dtypes(include='object').columns)>0:
      self.dummy = Dummify(self.target)
      data = self.dummy.fit_transform(dataset)
    else:
      data = dataset.copy()

    # reduce data size for faster processing
    # try:
    #   data = data.astype('float16')
    # except:
    #   None

    self.knn.fit(data.drop(self.target,axis=1))
    knn_predict = self.knn.predict(data.drop(self.target,axis=1))
    
    self.iso.fit(data.drop(self.target,axis=1))
    iso_predict = self.iso.predict(data.drop(self.target,axis=1))

    self.pca.fit(data.drop(self.target,axis=1))
    pca_predict = self.pca.predict(data.drop(self.target,axis=1))

    data['knn'] = knn_predict
    data['iso'] = iso_predict
    data['pca'] = pca_predict

    data['vote_outlier'] =  data['knn']  + data['iso'] + data['pca'] 
    self.outliers = data[data['vote_outlier']==3]
    
    return(dataset[[True if i not in self.outliers.index else False for i in dataset.index]])


#____________________________________________________________________________________________________________________________________________________________________
# Column Name cleaner transformer
class Clean_Colum_Names(BaseEstimator,TransformerMixin):
  '''
    - Cleans special chars that are not supported by jason format
  '''

  def __init__(self):
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data= dataset.copy()
    data.columns= data.columns.str.replace('[,}{\]\[\:\"\']','')
    return(data)

  def fit_transform(self,dataset,y=None):
    data= dataset.copy()
    data.columns= data.columns.str.replace('[,}{\]\[\:\"\']','')
    return(data)
#__________________________________________________________________________________________________________________________________________________________________________
# Clustering entire data
class Cluster_Entire_Data(BaseEstimator,TransformerMixin):
  '''
    - Applies kmeans clustering to the entire data set and produce clusters
    - Highly recommended to run the DataTypes_Auto_infer class first
      Args:
          target_variable: target variable (integer or numerical only)
          check_clusters_upto: to determine optimum number of kmeans clusters, set the uppler limit of clusters
  '''

  def __init__(self, target_variable, check_clusters_upto=20):
    self.target = target_variable
    self.check_clusters = check_clusters_upto +1
    
    

  def fit(self,data,y=None):
    return(None)

  def transform(self,data,y=None):
    # first convert to dummy
    if len(data.select_dtypes(include='object').columns)>0:
      data_t1 = self.dummy.transform(data)
    else:
      data_t1 = data.copy()

    # # now make PLS 
    data_t1 = self.pls.transform(data_t1)
    # now predict with the clustes
    predict =pd.DataFrame(self.k_object.predict(data_t1),index=data.index)
    data['data_cluster'] = predict
    data['data_cluster'] = data['data_cluster'].astype('object')
    return(data)

  def fit_transform(self,data,y=None):
    # first convert to dummy (if there are objects in data set)
    if len(data.select_dtypes(include='object').columns)>0:
      self.dummy = Dummify(self.target)
      data_t1 = self.dummy.fit_transform(data)
    else:
      data_t1 = data.copy()
    
    # now make PLS 
    self.pls = PLSRegression(n_components=len(data_t1.columns)-1)
    data_t1 = self.pls.fit_transform(data_t1.drop(self.target,axis=1),data_t1[self.target])[0]
    
    # we are goign to make a place holder , for 2 to 20 clusters
    self.ph = pd.DataFrame(np.arange(2,self.check_clusters,1), columns= ['clusters']) 
    self.ph['Silhouette'] = float(0)
    self.ph['calinski'] = float(0)

    # Now start making clusters
    for k in self.ph.index:
        c =  self.ph['clusters'][k]
        self.k_object =  cluster.KMeans(n_clusters= c,init='k-means++',precompute_distances='auto',n_init=10,random_state=42)
        self.k_object.fit(data_t1)
        self.ph.iloc[k,1] = metrics.silhouette_score(data_t1,self.k_object.labels_)
        self.ph.iloc[k,2] = metrics.calinski_harabaz_score(data_t1,self.k_object.labels_)
    
    # now standardize the scores and make a total column
    m = MinMaxScaler((-1,1))
    self.ph['calinski'] = m.fit_transform(np.array(self.ph['calinski']).reshape(-1,1))
    self.ph['Silhouette'] = m.fit_transform(np.array(self.ph['Silhouette']).reshape(-1,1))
    self.ph['total']= self.ph['Silhouette'] + self.ph['calinski']
    # sort it by total column and take the first row column 0 , that would represent the optimal clusters
    self.clusters = int(self.ph[self.ph['total'] == max(self.ph['total'])]['clusters'])
    # Now make the final cluster object
    self.k_object =  cluster.KMeans(n_clusters= self.clusters,init='k-means++',precompute_distances='auto',n_init=10,random_state=42)
    # now do fit predict
    predict =pd.DataFrame(self.k_object.fit_predict(data_t1),index=data.index)
    data['data_cluster'] = predict
    data['data_cluster'] = data['data_cluster'].astype('object')

    return(data)

#_________________________________________________________________________________________________________________________________________
class Fix_multicollinearity(BaseEstimator,TransformerMixin):
  """
          Fixes multicollinearity between predictor variables , also considering the correlation between target variable.
          Only applies to regression or two class classification ML use case
          Takes numerical and one hot encoded variables only
            Args:
              threshold (float): The utmost absolute pearson correlation tolerated beyween featres from 0.0 to 1.0
              target_variable (str): The target variable/column name
              correlation_with_target_threshold: minimum absolute correlation required between every feature and the target variable , default 1.0 (0.0 to 1.0)
              correlation_with_target_preference: float (0.0 to 1.0), default .08 ,while choosing between a pair of features w.r.t multicol & correlation target , this gives 
              the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
              A value of .5 means both measure have equal say 
  """
  # mamke a constructer
  
  def __init__ (self,threshold,target_variable,correlation_with_target_threshold= 0.0,correlation_with_target_preference=1.0):
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
      
    #global data1
    self.data1 = data.copy()
    # try:
    #   self.data1 = self.data1.astype('float16')
    # except:
    #   None
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
  def transform(self,dataset,y=None):
    '''
        Args:f
            data = takes preprocessed data frame
        Returns:
            data frame
    '''
    data = dataset.copy()
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

#____________________________________________________________________________________________________________________________________
# reduce feature space
class Reduce_Dimensions_For_Supervised_Path(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return same DF with different types of dimensionality reduction modles (pca_liner , pca_kernal, tsne , pls, incremental)
    - except pca_liner, every other method takes integer as number of components 
    - only takes numeric variables (float & One Hot Encoded)
    - it is intended to solve supervised ML usecases , such as classification / regression
  '''

  def __init__(self, target, method='pca_liner', variance_retained_or_number_of_components=.99, random_state=42):
    self.target= target
    self.variance_retained = variance_retained_or_number_of_components
    self.random_state= random_state
    self.method = method
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    if self.method in ['pca_liner' , 'pca_kernal', 'tsne' , 'pls', 'incremental']:
      data_pca = self.pca.transform(dataset)
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      return(data_pca)
    else:
      return(dataset)

  def fit_transform(self,dataset,y=None):

    if self.method == 'pca_liner':
      self.pca = PCA(self.variance_retained,random_state=self.random_state)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'pca_kernal': # take number of components only
      self.pca = KernelPCA(self.variance_retained,kernel='rbf',random_state=self.random_state,n_jobs=-1)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'pls': # take number of components only
      self.pca = PLSRegression(self.variance_retained,scale=False)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1),dataset[self.target])[0] 
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'tsne': # take number of components only
      self.pca = TSNE(self.variance_retained,random_state=self.random_state)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'incremental': # take number of components only
      self.pca = IncrementalPCA(self.variance_retained)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    else:
      return(dataset)


#___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one
def Preprocess_Path_One(train_data,target_variable,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=True,
                                imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                apply_zero_nearZero_variance = False,
                                club_rare_levels = False, rara_level_threshold_percentage =0.05,
                                apply_binning=False, features_to_binn =[],
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                target_transformation= False,target_transformation_method ='bc',
                                remove_outliers = False, outlier_contamination_percentage= 0.01,
                                remove_multicollinearity = False, maximum_correlation_between_features= 0.90,
                                cluster_entire_data= False, range_of_clusters_to_try=20, 
                                apply_pca = False , pca_method = 'pca_liner',pca_variance_retained_or_number_of_components =.99 ,
                                random_state=42

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Drop categorical variables that have zero variance or near zero variance
      - 4) Club categorical variables levels togather as a new level (other_infrequent) that are rare / at the bottom 5% of the variable distribution
      - 5) Generate sub features from time feature such as 'month','weekday',is_month_end','is_month_start' & 'hour'
      - 6) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile,maxabs,robust) , including option to transform target variable
      - 7) Apply binning to continious variable when numeric features are provided as a list 
      - 8) Detect & remove outliers using isolation forest, knn and PCA
      - 9) Apply clusters to segment entire data
      -10) One Hot / Dummy encoding
      -11) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      -12) Fix multicollinearity
      -13) Apply diamension reduction techniques such as pca_liner, pca_kernal, incremental, tsne & pls
          - except for pca_liner, all other method only takes number of component (as integer) i.e no variance explaination metohd available  
  '''
  global c2, subcase
  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  if ml_usecase is None:
    if ( ( (c1) & (c2) ) | (c3)   ):
      ml_usecase ='classification'
    else:
      ml_usecase ='regression'
  
  if ((len(train_data[target_variable].unique()) > 2) and (ml_usecase != 'regression')):
    subcase = 'multiclass'
  else:
    subcase = 'binary'
  
  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,categorical_features=categorical_features,numerical_features=numerical_features,time_features=time_features,features_todrop=features_todrop,display_types=display_types)

  
  # for imputation
  if imputation_type == "simple imputer":
    global imputer
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,categorical_strategy=categorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,categorical_strategy=categorical_imputation_strategy,target_variable=target_variable)
  
  # for zero_near_zero
  if apply_zero_nearZero_variance == True:
    global znz
    znz = Zroe_NearZero_Variance(target=target_variable)
  else:
    znz = Empty()

  # for rare levels clubbing:
  
  if club_rare_levels == True:
    global club_R_L
    club_R_L = Catagorical_variables_With_Rare_levels(target=target_variable,threshold=rara_level_threshold_percentage)
  else:
    club_R_L= Empty()

  # binning 
  if apply_binning == True:
    global binn
    binn = Binning(features_to_discretize=features_to_binn)
  else:
    binn = Empty()

  # scaling & power transform
  if scale_data == True:
    global scaling 
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method,random_state_quantile=random_state)
  else: 
    scaling = Empty()
  
  if Power_transform_data== True:
    global P_transform
    P_transform = Scaling_and_Power_transformation(target=target_variable,function_to_apply=Power_transform_method,random_state_quantile=random_state)
  else:
    P_transform= Empty()
  
  # target transformation
  if ((target_transformation == True) and (ml_usecase == 'regression')):
    global pt_target
    pt_target = Target_Transformation(target=target_variable,function_to_apply=target_transformation_method)
  else:
    pt_target = Empty()


  # for Time Variables
  global feature_time
  feature_time = Make_Time_Features()
  global dummy
  dummy = Dummify(target_variable)

  # remove putliers
  if remove_outliers == True:
    global rem_outliers
    rem_outliers= Outlier(target=target_variable,contamination=outlier_contamination_percentage,random_state=random_state )
  else:
    rem_outliers = Empty()
  
  # cluster all data:
  if cluster_entire_data ==True:
    global cluster_all
    cluster_all = Cluster_Entire_Data(target_variable=target_variable,check_clusters_upto=range_of_clusters_to_try)
  else:
    cluster_all = Empty()
  
  # clean column names for special char
  clean_names =Clean_Colum_Names()
  
  # removing multicollinearity
  if remove_multicollinearity == True and subcase != 'multiclass':
    global fix_multi
    fix_multi = Fix_multicollinearity(target_variable=target_variable,threshold=maximum_correlation_between_features)
  else:
    fix_multi = Empty()

  
  # apply pca
  if apply_pca == True:
    global pca
    pca = Reduce_Dimensions_For_Supervised_Path(target=target_variable,method = pca_method ,variance_retained_or_number_of_components=pca_variance_retained_or_number_of_components, random_state=random_state)
  else:
    pca= Empty()

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('znz',znz),
                 ('club_R_L',club_R_L),
                 ('feature_time',feature_time),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('pt_target',pt_target),
                 ('binn',binn),
                 ('rem_outliers',rem_outliers),
                 ('cluster_all',cluster_all),
                 ('dummy',dummy),
                 ('clean_names',clean_names),
                 ('fix_multi',fix_multi),
                 ('pca',pca)
                 ])
  
  if test_data is not None:
    return(pipe.fit_transform(train_data),pipe.transform(test_data))
  else:
    return(pipe.fit_transform(train_data))



# ______________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one_unsupervised
def Preprocess_Path_Two(train_data,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=False,
                                imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                apply_zero_nearZero_variance = False,
                                club_rare_levels = False, rara_level_threshold_percentage =0.05,
                                apply_binning=False, features_to_binn =[],
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                remove_outliers = False, outlier_contamination_percentage= 0.01,
                                remove_multicollinearity = False, maximum_correlation_between_features= 0.90,  
                                apply_pca = False , pca_method = 'pca_liner',pca_variance_retained_or_number_of_components =.99 , 
                                random_state=42

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - THIS IS BUILt FOR UNSUPERVISED LEARNING , FOLLOWES SAME PATH AS Path_One
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Drop categorical variables that have zero variance or near zero variance
      - 4) Club categorical variables levels togather as a new level (other_infrequent) that are rare / at the bottom 5% of the variable distribution
      - 5) Generate sub features from time feature such as 'month','weekday',is_month_end','is_month_start' & 'hour'
      - 6) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile,maxabs,robust) , including option to transform target variable
      - 7) Apply binning to continious variable when numeric features are provided as a list 
      - 8) Detect & remove outliers using isolation forest, knn and PCA
      - 9) One Hot / Dummy encoding
      -10) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      -11) Fix multicollinearity
      -12) Apply diamension reduction techniques such as pca_liner, pca_kernal, incremental, tsne & pls
          - except for pca_liner, all other method only takes number of component (as integer) i.e no variance explaination metohd available  
  '''
  
  # just make a dummy target variable
  target_variable = 'dummy_target'
  train_data[target_variable] = 2
  # just to add diversified values to target
  train_data.loc[0:3,target_variable] = 3
 
  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  # dummy usecase
  ml_usecase ='regression'
  

  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,categorical_features=categorical_features,numerical_features=numerical_features,time_features=time_features,features_todrop=features_todrop,display_types=display_types)

  
  # for imputation
  global imputer
  if imputation_type == "simple imputer":
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,categorical_strategy=categorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,categorical_strategy=categorical_imputation_strategy,target_variable=target_variable)
  
  # for zero_near_zero
  global znz
  if apply_zero_nearZero_variance == True:
    znz = Zroe_NearZero_Variance(target=target_variable)
  else:
    znz = Empty()
 
  # for rare levels clubbing:
  global club_R_L
  if club_rare_levels == True:
    club_R_L = Catagorical_variables_With_Rare_levels(target=target_variable,threshold=rara_level_threshold_percentage)
  else:
    club_R_L= Empty()
  
  # binning 
  global binn

  if apply_binning == True:
    binn = Binning(features_to_discretize=features_to_binn)
  else:
    binn = Empty()
  
  # for scaling
  global scaling ,P_transform
  if scale_data == True:
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method,random_state_quantile=random_state)
  else: 
    scaling = Empty()
  
  if Power_transform_data== True:
    P_transform = Scaling_and_Power_transformation(target=target_variable,function_to_apply=Power_transform_method,random_state_quantile=random_state)
  else:
    P_transform= Empty()

  # for Time Variables
  global feature_time
  feature_time = Make_Time_Features()
  global dummy
  dummy = Dummify(target_variable)
  
  # remove putliers
  if remove_outliers == True:
    global rem_outliers
    rem_outliers= Outlier(target=target_variable,contamination=outlier_contamination_percentage,random_state=random_state )
  else:
    rem_outliers = Empty()

  # clean column names for special char
  clean_names =Clean_Colum_Names()

  # removing multicollinearity
  if remove_multicollinearity == True: 
    global fix_multi
    fix_multi = Fix_multicollinearity(target_variable=target_variable,threshold=maximum_correlation_between_features,correlation_with_target_preference=0.0)
  else:
    fix_multi = Empty()

  # apply pca
  global pca
  if apply_pca == True:
    pca = Reduce_Dimensions_For_Supervised_Path(target=target_variable,method = pca_method ,variance_retained_or_number_of_components=pca_variance_retained_or_number_of_components, random_state=random_state)
  
  else:
    pca= Empty()

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('znz',znz),
                 ('club_R_L',club_R_L),
                 ('feature_time',feature_time),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('binn',binn),
                 ('rem_outliers',rem_outliers),
                 ('dummy',dummy),
                 ('clean_names',clean_names),
                 ('fix_multi',fix_multi),
                 ('pca',pca)
                 ])
  
  if test_data is not None:
    train_t = pipe.fit_transform(train_data)
    test_t = pipe.transform(test_data)
    return(train_t.drop(target_variable,axis=1),test_t)
  else:
    train_t = pipe.fit_transform(train_data)
    return(train_t.drop(target_variable,axis=1))
