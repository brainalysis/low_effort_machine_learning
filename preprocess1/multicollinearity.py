from sklearn.base import BaseEstimator , TransformerMixin

class fix_multicollinearity(BaseEstimator,TransformerMixin):
    """
            Fixes multicollinearity between predictor variables , also considering the correlation between target variable.
            Only applies to regression or two class classification ML use case
              Args:
                threshold (numeric): The utmost pearson correlation tolerated beyween featres
                target_variable (str): The target variable/column name 
    """
    # mamke a constructer
    
    def __init__ (self,threshold,target_variable):
        self.threshold = threshold
        self.target_variable = target_variable
    
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
        data1 = data.copy()
        # make an correlation db with abs correlation db
        corr_matrix = abs(data1.corr())

        # for every diagonal value, make it Nan
        corr_matrix.values[tuple([np.arange(corr_matrix.shape[0])]*2)] = np.NaN
        
        # Now Calculate the average correlation of every feature with other, and get a pandas data frame
        avg_cor = pd.DataFrame(corr_matrix.mean())
        avg_cor['feature']= avg_cor.index
        avg_cor.reset_index(drop=True, inplace=True)
        avg_cor.columns =  ['avg_cor','features']
        
        # Calculate the correlation with the target
        targ_cor = pd.DataFrame(corr_matrix[self.target_variable].dropna())
        targ_cor['feature']= targ_cor.index
        targ_cor.reset_index(drop=True, inplace=True)
        targ_cor.columns =  ['target_variable','features']
        
        # Now, add a column for variable name and drop index
        corr_matrix['column'] = corr_matrix.index
        corr_matrix.reset_index(drop=True,inplace=True)
        
        # now we need to melt it , so that we can corelation pair wise , with two columns 
        cols =corr_matrix.column
        melt = corr_matrix.melt(id_vars= ['column'],value_vars=cols).sort_values(by='value',ascending=False).dropna()

        # now bring in the avg correlation for first of the pair
        merge = pd.merge(melt,avg_cor,left_on='column',right_on='features').drop('features',axis=1)

        # now bring in the avg correlation for second of the pair
        merge = pd.merge(merge,avg_cor,left_on='variable',right_on='features').drop('features',axis=1)
  
        # now bring in the target correlation for first of the pair
        merge = pd.merge(merge,targ_cor,left_on='column',right_on='features').drop('features',axis=1)

        # now bring in the avg correlation for second of the pair
        merge = pd.merge(merge,targ_cor,left_on='variable',right_on='features').drop('features',axis=1)

        # sort and save
        merge = merge.sort_values(by='value',ascending=False)

        # we need to now eleminate all the pairs that are actually duplicate e.g cor(x,y) = cor(y,x) , they are the same , we need to find these and drop them
        merge['all_columns'] = merge['column'] + merge['variable']

        # this puts all the coresponding pairs of features togather , so that we can only take one, since they are just the duplicates
        merge['all_columns'] = [sorted(i) for i in merge['all_columns'] ]

        # now sort by new column
        merge = merge.sort_values(by='all_columns')

        # take every second colums
        merge = merge.iloc[::2, :]

        # make a ranking column to eliminate features
        merge['rank_x'] = round((merge['avg_cor_y']- merge['avg_cor_x']) + (merge['target_variable_x'] - merge['target_variable_y']),6) # round it to 6 digits

        ## Now there will be rows where the rank will be exactly zero, these is where the value (corelartion between features) is exactly one ( like price and price^2)
        ## so in that case , we can simply pick one of the variable
        # but since , features can be in either column, we will drop one column (say 'column') , only if the feature is not in the second column (in variable column)
        # both equations below will return the list of columns to drop from here 
        # this is how it goes

        ## For the portion where correlation is exactly one !
        one = merge[merge['rank_x']==0]

        #[i for i in pd.unique(small['column']) if i not in pd.unique(small['variable'])]
        to_drop =(list(set(one['column'])-set(one['variable'])))

        ## now we are to treat where rank is not Zero and Value (corelation) is greater than a specific threshold
        non_zero = merge[(merge['rank_x']!= 0.0) & (merge['value'] >= self.threshold)]

        # pick the column to delete
        non_zero_list = list(np.where(non_zero['rank_x'] < 0 , non_zero['column'], non_zero['variable']))

        # add two list
        to_drop = to_drop + non_zero_list

        #make sure that target column is not a part of the list
        try:
            to_drop.remove(self.target_variable)
        except:
            to_drop
        
        self.to_drop = to_drop

    # now Transform
    def transform(self,data,y=None):
        '''
            Args:
                data = takes preprocessed data frame
            Returns:
                data frame
        '''
        data.drop(self.to_drop,axis=1,inplace=True)
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
        
        
      