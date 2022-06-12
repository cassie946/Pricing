import pandas as pd
df = pd.read_csv('candidate_1.tab.txt', delimiter='\t')
#print(df)
#about data: consulting is choosing candidate

#print(df.columns)
#print(df.shape)
#print(df.describe())
#print(df['religion'].unique())
#print(df.dtypes)
'''
Index(['MgrID', 'education', 'religion', 'research_area', 'professional',
       'pricing_group', 'race', 'age_group', 'gender', 'selected', 'rating'],
      dtype='object')
(3466, 11)
             MgrID    education  ...     selected       rating
count  3466.000000  3466.000000  ...  3466.000000  3456.000000
mean    455.171379     1.498846  ...     0.500000     0.509838
std     252.867078     0.500071  ...     0.500072     0.246864
min       7.000000     1.000000  ...     0.000000     0.000000
25%     238.000000     1.000000  ...     0.000000     0.333333
50%     453.000000     1.000000  ...     0.500000     0.500000
75%     669.000000     2.000000  ...     1.000000     0.666667
max     900.000000     2.000000  ...     1.000000     1.000000

[8 rows x 9 columns]
['others' 'Chrisitan' 'buddist' 'hindu' 'muslim' 'catholic']
MgrID              int64
education          int64
religion          object
research_area      int64
professional       int64
pricing_group      int64
race               int64
age_group          int64
gender            object
selected           int64
rating           float64
dtype: object

'''

#看有没有missing value
print(df.isnull().sum())
df_input=df[['education', 'religion', 'research_area', 'professional',
       'pricing_group', 'race', 'age_group', 'gender']]
print(df_input)
'''
MgrID             0
education         0
religion          0
research_area     0
professional      0
pricing_group     0
race              0
age_group         0
gender            0
selected          0
rating           10
     education   religion  research_area  ...  race  age_group  gender
0             1     others              3  ...     1          6  female
1             2  Chrisitan              1  ...     6          4    male
2             1    buddist              5  ...     2          5  female
3             2      hindu              5  ...     1          6    male
4             2  Chrisitan              2  ...     2          2  female
...         ...        ...            ...  ...   ...        ...     ...
3461          2    buddist              3  ...     3          4    male
3462          1     muslim              6  ...     5          4    male
3463          2      hindu              4  ...     1          2    male
3464          2     others              3  ...     4          2    male
3465          2      hindu              4  ...     4          1    male
[3466 rows x 8 columns]
'''
#rating 有10个missing value
#how to deal with rating missing value : replace with mean
#只有10个 可以ignore

import seaborn as sns
import  matplotlib. pyplot as plt

#sns.countplot(data=df_input,x='gender', hue='religion' )
#plt.show()

print(pd.get_dummies(data=df_input,columns=df_input.columns))
'''
      education_1  education_2  ...  gender_female  gender_male
0               1            0  ...              1            0
1               0            1  ...              0            1
2               1            0  ...              1            0
3               0            1  ...              0            1
4               0            1  ...              1            0
...           ...          ...  ...            ...          ...
3461            0            1  ...              0            1
3462            1            0  ...              0            1
3463            0            1  ...              0            1
3464            0            1  ...              0            1
3465            0            1  ...              0            1
[3466 rows x 40 columns]
'''

#label encoding
#onehot encoding

#fit with ols model
import statsmodels.api as sm

olsModel= sm.OLS(df['selected'],pd.get_dummies(data=df_input,columns=df_input.columns))
res =olsModel.fit()
print(res.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               selected   R-squared:                       0.092
Model:                            OLS   Adj. R-squared:                  0.083
Method:                 Least Squares   F-statistic:                     10.82
Date:                Sun, 12 Jun 2022   Prob (F-statistic):           1.77e-51
Time:                        17:26:42   Log-Likelihood:                -2349.0
No. Observations:                3466   AIC:                             4764.
Df Residuals:                    3433   BIC:                             4967.
Df Model:                          32                                         
Covariance Type:            nonrobust                                         
======================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------
education_1         2.366e+11   3.71e+11      0.638      0.523    -4.9e+11    9.63e+11
education_2         2.366e+11   3.71e+11      0.638      0.523    -4.9e+11    9.63e+11
religion_Chrisitan -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
religion_buddist   -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
religion_catholic  -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
religion_hindu     -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
religion_muslim    -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
religion_others    -8.943e+11    1.4e+12     -0.638      0.523   -3.64e+12    1.85e+12
research_area_1     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
research_area_2     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
research_area_3     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
research_area_4     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
research_area_5     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
research_area_6     1.966e+10   3.08e+10      0.638      0.523   -4.07e+10       8e+10
professional_1      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
professional_2      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
professional_3      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
professional_4      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
professional_5      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
professional_6      9.338e+10   1.46e+11      0.638      0.523   -1.93e+11     3.8e+11
pricing_group_1     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
pricing_group_2     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
pricing_group_3     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
pricing_group_4     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
pricing_group_5     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
pricing_group_6     8.327e+10    1.3e+11      0.638      0.523   -1.72e+11    3.39e+11
race_1              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
race_2              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
race_3              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
race_4              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
race_5              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
race_6              1.096e+11   1.72e+11      0.638      0.523   -2.27e+11    4.46e+11
age_group_1         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
age_group_2         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
age_group_3         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
age_group_4         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
age_group_5         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
age_group_6         7.264e+10   1.14e+11      0.638      0.523    -1.5e+11    2.96e+11
gender_female       2.792e+11   4.37e+11      0.638      0.523   -5.78e+11    1.14e+12
gender_male         2.792e+11   4.37e+11      0.638      0.523   -5.78e+11    1.14e+12
==============================================================================
Omnibus:                    16514.277   Durbin-Watson:                   2.874
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              390.654
Skew:                          -0.013   Prob(JB):                     1.48e-85
Kurtosis:                       1.355   Cond. No.                     7.99e+15
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.09e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
'''