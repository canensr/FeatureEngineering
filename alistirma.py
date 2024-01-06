
############# Feature Engineering on Diabetes Dataset #############

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)


## Task 1 : Keşifçi Veri Analizi

# Step 1:  Genel resmi inceleyiniz

data = pd.read_csv("Feature_Engineering/datasets/diabetes.csv")
df = data.copy()
df.head()

df.shape
df.describe().T
df.isnull().sum() / df.shape[0] * 100
df.info()
df.dtypes

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

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

cat_cols
num_cols

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
def get_stats(dataframe, col):
    return print("------ İlk 5 Satır ------ \n", dataframe[col].head(), "\n", \
                 "------ Sahip olduğu Değer Sayısı ------ \n", dataframe[col].value_counts(), "\n", \
                 "------ Toplam Gözlem Sayısı ------ \n",  dataframe[col].shape, "\n", \
                 "------ Değişken Tipleri ------ \n", dataframe[col].dtypes, "\n", \
                 "------ Toplam Null Değer Sayısı ------ \n", dataframe[col].isnull().sum(), "\n", \
                 "------ Betimsel İstatistik ------ \n", dataframe[col].describe().T
                 )

get_stats(df, cat_cols)
get_stats(df, num_cols)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###############################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, True)

# Adım 4 : Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre Numerik değişkenlerin ortalaması)
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
# Hedef değişkenimiz zaten outcome = kategorik değişken


# Numerik değişkenlere göre hedef değişkenin ortalamas
# df.groupby(num_cols)[cat_cols].mean()  # saçma bi çıktı verdi

# hedef değişkene göre Numerik değişkenlerin ortalaması
df.groupby(cat_cols)[num_cols].mean()

# Adım 5: Aykırı gözlem analizi yapınız.

# baskılama
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
        return False   # var mı? yok mu? sorusuna bool dönmesi lazım (True veya False)

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



# Adım 6: Eksik gözlem analizi yapınız.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)


# Adım 7: Korelasyon analizi yapınız
df.corr().sort_values("Outcome", ascending=False) \
    .drop("Outcome", axis=0)

sns.heatmap(df.corr(), annot=True)
plt.show(block=True)
# daha iyi sonuç alabilmek adına, yapılan gözlemden kendisini(Outcome) çıkardım


## Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

# ilk olarak, aykırı değer var mı? yok mu? kontrolü



# for col in num_cols:
#     print(df[col].value_counts())

# 0 -> NAN




# Adım 2: Yeni değişkenler oluşturunuz.


# Adım 3: Encoding işlemlerini gerçekleştiriniz.

# yeni oluşturduğum kategorik değişkenleri gözlemlerken, teker teker value_counts'larına bakmak yerine
# fonksiyon yazmayı tercih ettim




# One Hot Encoding
# eşsiz değer sayısı 2 den fazla olanların sayısı 10 dan küçük veya 10'a eşitse getir




# Adım 4: Numerik değişkenler için standartlaştırma yapınız.


# standartlaştırma



# Adım 5: Model oluşturunuz.


