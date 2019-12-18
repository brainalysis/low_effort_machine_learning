import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator , TransformerMixin

class Fix_multicollinearity(BaseEstimator,TransformerMixin):
  """
          Fixes multicollinearity between predictor variables , also considering the correlation between target variable.
          Only applies to regression or two class classification ML use case
            Args:
              threshold (float): The utmost absolute pearson correlation tolerated beyween featres from 0.0 to 1.0
              target_variable (str): The target variable/column name
              correlation_with_target_threshold: minimum absolute correlation required between every feature and the target variable , default 1.0 (0.0 to 1.0)
  """
  # mamke a constructer
  
  def __init__ (self,threshold,target_variable,correlation_with_target_threshold= 0.0):
      self.threshold = threshold
      self.target_variable = target_variable
      self.correlation_with_target_threshold= correlation_with_target_threshold
  
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
    # make an correlation db with abs correlation db
    # self.data_c = self.data1.T.drop_duplicates()
    # self.data1 = self.data_c.T
    self.corr_matrix = abs(self.data1.corr())

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
    self.merge['rank_x'] = round((self.merge['avg_cor_y']- self.merge['avg_cor_x']) + (self.merge['target_variable_x'] - self.merge['target_variable_y']),6) # round it to 6 digits
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
      self.to_drop_taret_correlation = data.drop(self.to_drop,axis=1).corr()[self.target_variable].abs()
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
        Args:
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