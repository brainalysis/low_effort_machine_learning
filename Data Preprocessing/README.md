# PreProcess1
## Super Easy Way of PreProcessing your Data!

Good News! 

Preprocess1 simplifies the preprocessing steps that are some time essential for ML/modelling, such as imputations,
one hot encoding. There are over 20 preprocessing steps available. A summary of the options is below:

- Auto infer data types 
- Impute (simple or with surrogate columns)
- Ordinal Encoder
- Drop categorical variables that have zero variance or near-zero variance
- Club categorical variables levels together as a new level (other_infrequent) that are rare / at the bottom 5% of the variable  
  distribution
- Club unseen levels in test dataset with most/least frequent levels in train dataset 
- Reduce high cardinality in categorical features using clustering or counts
- Generate sub-features from time feature such as 'month','weekday',is_month_end','is_month_start' & 'hour'
- Group features by calculating min, max, mean, median & sd of similar features
- Make nonlinear features (polynomial, sin, cos & tan)
- Scales & Power Transform (zscore,minmax,yeo-johnson,quantile,maxabs,robust) , including option to transform target variable
- Apply binning to variables when numeric features are provided as a list 
- Detect & remove outliers using isolation forest, KNN and PCA
- Apply clusters to segment entire data
- One Hot / Dummy encoding
- Remove special characters from column names such as commas, square brackets etc. to make it compatible with Jason dependent models
- Feature Selection through Random Forest, LightGBM and Pearson Correlation
- Fix multicollinearity
- Feature Interaction (DFS), multiply, divided, add and subtract features
- Apply dimension reduction techniques such as pca_liner, pca_kernal, incremental or  Tsne. 
  except for pca_liner, all other methods only take the number of components (as integer)
  i.e no variance explanation method available
  
You can install the library as 

```python
pip install preprocess1
from preprocess1 import toolkit as t
```

Although one can use the methods individually (by calling the respective class) , such as: 

```python
binn = t.Binning(['feature_tobin'])
binned_data = binn.fit_transform(training_data)
binned_new_data = binn.transform(test_data)
```

However, there is more power to it. We have made pre-built complete pipelines to deploy all sorts of preprocessing transformers. Path1 is for supervised ML, and Path2 is for unsupervised ML problems. Below is how you use it:

```python
# apply the path to the training dataset while clubbing rare categorical levels & scaling numerical features
# Imputation & One Hot Encoding is automatically applied
data_training_transformed = t.Preprocess_Path_One(training_data, 'target_column', club_rare_levels = True, scale_data= True)
# apply the pipeline to the test data set
data_test_transformed = pipe.fit_transform(test_data)
```

You can find more information under the docstring of each class/function. Enjoy coding! 
Please share your ideas, suggestions and critique with me.


## License

Copyright 2019-2020 Fahad Akbar <fahad.akbar@gmail.com>


