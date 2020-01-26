# Multicollinearity:
This has been an issue for some Machine Learning models, although few algorithms are not affected by multicollinearity, but at the least , reducing redundant features will make the model less expensive in terms of computational power.

This function looks to reduce feature space based on correlation between features and at the same time , looking at the correlation between features and target variable

A simple rule of thumb is , say your feature A and B are highly correlated , then we need to drop on of the features. We will drop the feature (say feature B)  that has the :
  
  1) More average correlation with all other variables in the rest of the data set
  2) Less correlation with Target , then the other Variable
  
Although I found it useful , yet simply running this code may not be efficient alone, because it does not consider the impact of feature interaction alone. I am developing another comprehensive preprocessing function that will take care of this issue. I will post that late on. For now, it is better to use this code once you have done your feature engineering / feature interactions

Also, this is supposed to work for regression and two class classification problems

In the end , please let me know if there are any glitches, room for improvements (i am pretty sure that there are many ) etc, after all , we all learn from each other’s mistakes  :-)

Thanks
Fahad

# Instructions:

Function takes following arguments

 1) Data : Panda's Data Frame is required
 2) Threshold: The minimum level of correlation that you want to see between variables , between 0 - 1 , absolute values
 3) Target : specify the target column (y)
 4) correlation_with_target_threshold: minimum absolute correlation required between every feature and the target variable , default 1.0 (0.0 to 1.0)
 5) correlation_with_target_preference: float (0.0 to 1.0), default .08 ,while choosing between a pair of features w.r.t multicol & correlation target , this gives 
    the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
    A value of .5 means both measure have equal say

# Preprocessing Toolkit

This portion attempts to simplyfy the preprocessing steps that are some time essential for ML / modeling , such as imputations,
one hot encoding and so on. Predetermined preprocessing 'paths' are available under preprocess1.toolkit . See docstring for more details