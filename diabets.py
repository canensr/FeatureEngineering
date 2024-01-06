import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Feature_Engineering/datasets/diabetes.csv")
df.head()
df.isnull().sum()
df.describe().T
df.info()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols
cat_cols

df.describe().T
df.info()
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#4
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


#5
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#aykırı değer var mı? var.
for col in num_cols:
    print(col, check_outlier(df, col))

#baskılama işlemi
for col in num_cols:
    replace_with_thresholds(df, col)

#baskılandı mı? tekrar aykırı değer var mı kontrolü
for col in num_cols:
    print(col, check_outlier(df, col))

#6
df.isnull().sum()


#7
correlation_matrix = df.corr()

# Korelasyon matrisini görselleştirin
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Diabetes Veri Seti Korelasyon Matrisi')
plt.show()


#görev2 adım 1







#görev2 adım2
df.loc[df['Age'] <= 30, 'NEW_AGE_CAT'] = 'young-age'
df.loc[(df['Age'] > 30) & (df['Age'] < 50), 'NEW_AGE_CAT'] = 'middle-age'
df.loc[df['Age'] > 50, 'NEW_AGE_CAT'] = 'old-age'
df.groupby('NEW_AGE_CAT').agg({'Outcome': 'mean'})

# RATIO_BMI/BP
df['RATIO_BMI/BP'] = df['BMI'] / df['BloodPressure']
df.groupby('Outcome').agg({'RATIO_BMI/BP': ['mean', 'count']})

# GLUCOSE_LEVEL
df['Glucose'].describe([0.25, 0.50, 0.75]).T
df.groupby('Outcome').agg({'Glucose': 'mean'})
df.loc[df['Glucose'] <= 99.750, 'GLUCOSE_LEVEL'] = 'Low'
df.loc[(df['Glucose'] > 99.750) & (df['Glucose'] <= 140.250), 'GLUCOSE_LEVEL'] = 'Mid'
df.loc[df['Glucose'] > 140.250, 'GLUCOSE_LEVEL'] = 'High'
df.head()

# PREG_AGE_SCORE
df['PREG_AGE_SCORE'] = df['Pregnancies'] * df['Age']
df.groupby('Outcome').agg({'PREG_AGE_SCORE': 'mean'})

# INS/SKT
df['INS/SKT'] = df['Insulin'] / df['SkinThickness']
df.groupby('Outcome').agg({'INS/SKT': 'mean'})

# AGE_BMI_div_SKT
df["AGE_BMI_div_SKT"] = (df['Age'] * df['BMI']) / df['SkinThickness']
df.groupby('Outcome').agg({'AGE_BMI_div_SKT': ['mean', 'count']})

# GLUCOSE_div_BP
df['GLUCOSE_div_BP'] = df['Glucose'] / df['BloodPressure']
df.groupby('Outcome').agg({'GLUCOSE_div_BP': 'mean'})

# BMI_HALF
df["BMI"].describe().T
df.loc[df["BMI"] <= 32.400, 'BMI_HALF'] = 'LOW_BMI'
df.loc[df["BMI"] > 32.400, 'BMI_HALF'] = 'HIGH_BMI'
df.groupby('BMI_HALF').agg({'Outcome': 'mean'})
##########################################3


















