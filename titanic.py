# import pandas and numpy libraries
import pandas as pd
import numpy as np

#import data
path = r"C:\Users\agiop\Documents\Towards Impossible\Programming for Data Science\Python\Python Scripts\Titanic\titanic.csv"
titanic = pd.read_csv( path )

# inspect data
titanic.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.7+ KB

# calculate basic descriptive statistics
titanic.describe()

# fixing missing values
import matplotlib.pyplot as plt

def countna(df):
    d_na = {}
    counts = []
    for col_id in range(0, df.shape[1]):
        na_count_id_col = df.iloc[:,col_id].isna().sum()
        counts.append(na_count_id_col)
        na_percentage_col_id = (na_count_id_col/df.shape[0])
        if (na_percentage_col_id>0):
            d_na[df.columns[col_id]] = (na_percentage_col_id*100)
    d_na_sort = sorted(d_na.items(), key = lambda i:i[1])
    names = [d_na_sort[i][0] for i in range(0, len(d_na_sort))]
    percentages = [d_na_sort[i][1] for i in range(0, len(d_na_sort))]
    # create plot
    fig = plt.figure()
    plt.bar(names,percentages)
    plt.title("Percentage of Missing Values")
    plt.xlabel("Name")
    plt.ylabel("Percentage (%)")
    return [names, counts, percentages, fig]

# column "embarked" has low cardinality and only two missing values. i fill them with the most frequent value.
value_frequencies = {i: (titanic["Embarked"] == i).sum() for i in titanic["Embarked"].dropna().unique()}
titanic["Embarked"].fillna(max(value_frequencies), inplace = True)

# column "Age" has some missing values, and its cardinality is large. here, we make use of other related
# information to every person, whose age is missing, in order to construct a linear regression model
# that will predict the unknown age of the person.

# first, what we would get, if we used the mean "Age" to impute the missing parts?
# the correlation matrix indicate that there exists a relationship between variables "Age" and "Pclass".
# although somewhat weak (-0.404651), it is the most prominent, hence we can base our imputation on it.
titanic[["Age", "Pclass", "SibSp", "Parch"]].corr().iloc[:,0]

# Age       1.000000
# Pclass   -0.404651
# SibSp    -0.243456
# Parch    -0.175185
# Name: Age, dtype: float64

# so, here, is what we would have if we used the corresponding mean "Age" for each "Pclass" in order to impute missing values.
titanic["Age"] = np.where(titanic["Age"].isna(),
                          np.select([titanic["Pclass"] == i for i in range(1, 4)],
                                    list(titanic.groupby("Pclass").mean().iloc[:,2])),
                          titanic["Age"])

# making some last changes in column "Age". converting dtype of "Age" from float64 to int8, will increase memory efficiency.
import math
titanic["Age"] = titanic["Age"].apply(lambda x: math.ceil(x) if x < 1 else round(x)).astype("int8")

# [TO DO] memory savings (%)?

# now, how can we compare this option to the one of using linear regression to impute the missing parts of "Age"?
# theoretically, the prediction error of the mean is simply the standard deviation. thus, we can use it as a benchamark.
# but is it BLUE (BLUE, here, means Best Linear Unbiased Estimator)? this, among others, means that standard deviation
# is minimum. for this to be the case, the distribution of "Age" must remain constant, when "Pclass" changes, which means
# that we have to check for normality and homoscedasticity accross populations.

import scipy.stats as stats
import itertools

def check_homoscedasticity_normality(data, variable, groupby, a = 0.05, check_normality = True):
    y = data[groupby].unique(); y.sort()
    x_groupby_y = [data.loc[data[groupby] == i, variable] for i in y]
    flag_homoscedasticity = True
    for i in list(itertools.combinations(y, 2)):
        if stats.bartlett(x_groupby_y[i[0]], x_groupby_y[i[1]])[1] < a:
            flag_homoscedasticity = False; break
    flag_normality = "-"
    if check_normality == True:
        flag_normality = True
        for i in y:
            if stats.shapiro(x_groupby_y[i])[1] < a:
                flag_normality = False; break
    print("Normality: " + str(flag_normality), ", ", "Homoscedasticity: " + str(flag_homoscedasticity))

# we may conclude that the distribution of "Age" does not remain constant while "Pclass" changes.
# normality condition is not satisfied, and heteroscedasticity is introduced. we shall not infer that standard error
# for non-missing values is an unbiased estimator of the prediction error. however, we may use it as a rough benchmark for
# measuring the predictive accuracy of the imputation using linear regression, even though being a misestimation.

baseline_error = stats.sem(titanic["Age"].dropna())

# comparing method: k-fold cross validation with resampling
data["fold"] = np.apply_along_axis(lambda x: np.repeat(x, 5), 0, np.arange(1, folds + 1))
np.where(np.array([1,2,3])==2)[0][0]

# data = dataframe,
# y = output variable (character), x = input variables (list of characters),
# k = number of folds (integer), r = number of resamplings (integer)
def kfoldcv_lr(data, y, x, k, r, impute = True):
    # if impute is True, then y has missing values, which have to be imputed.
    # in such case, proceed to k-fold partiotioning of the data without the rows, for which y is missing.
    # else, proceed in k-fold partitioning of the whole dataset.
    
    # create an equally partitioned array of k folds, where in each of them a different integer is stored, i.e.
    # [1st fold = (1, 1, ..., 1), 2nd fold = (2, 2, ..., 2), k-th fold = (k, k, ..., k)]
    fold = np.concatenate((
                           np.apply_along_axis(lambda x: np.repeat(x, data[y].dropna().shape[0] // k), 0, np.arange(1, k + 1)),
                           np.repeat(0, data[y].dropna().shape[0] % k)
                          )); data.insert(data.shape[1], "fold", fold)
    # empty list in which the cross validation errors will be stored
    cv_error = []
    for s in list(range(r)):
        # reshuffle samples
        data[fold] = np.random.shuffle(fold)
        # empty list in which the k-fold errors will be stored
        kfold_error = []
        for i in data[fold].unique():
            # define training and testing sets
            train_y, train_x = data.loc[data["fold"] != i, y], data.loc[data["fold"] != i, x]
            test_y, test_x = data.loc[((data["fold"] == i) or (data["fold"] == 0)), y], data.loc[((data["fold"] == i) or (data["fold"] == 0)), x]
            # MODEL
            
