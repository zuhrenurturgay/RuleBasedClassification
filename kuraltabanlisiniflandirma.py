import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

###GÖREV-1

def load_persona():
    df=pd.read_csv("datasets/persona.csv")
    return df

df=load_persona()

############# VERİ SETİ HİKAYESİ ###############
###Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve
#bu ürünleri satın alan kullanıcıların bazı demografik bilgilerini
#barındırmaktadır.
#Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir.
#Bunun anlamı tablo tekilleştirilmemiştir.
#Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

##### DEĞİŞKENLER #####

#PRICE – Müşterinin harcama tutarı
#SOURCE – Müşterinin bağlandığı cihaz türü
#SEX – Müşterinin cinsiyeti
#COUNTRY – Müşterinin ülkesi
#AGE – Müşterinin yaşı

##### İŞ PROBLEMİ #####
#Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları
#(persona) oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni
#gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



#SORU-2:Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].unique()
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

#SORU-3:Kaç unique PRICE vardır?

df["PRICE"].nunique()

#SORU-4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

#SORU-5:Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()
#yontem-2
df.groupby("COUNTRY")["PRICE"].count()

#SORU-6:Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df[["COUNTRY","PRICE"]].groupby("COUNTRY").agg("sum")

#SORU-7:SOURCE türlerine göre satış sayıları nedir?

df[["SOURCE","PRICE"]].groupby("SOURCE").agg("count")

#SORU-8:Ülkelere göre PRICE ortalamaları nedir?

df[["PRICE","COUNTRY"]].groupby("COUNTRY").agg({"PRICE":"mean"})

#SORU-9:SOURCE'lara göre PRICE ortalamaları nedir?

df[["PRICE","SOURCE"]].groupby("SOURCE").agg("mean")

#SORU-10:COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})


###GÖREV-2:COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

###GÖREV-3:Çıktıyı PRICE’a göre sıralayınız.

z=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

z.sort_values("PRICE", ascending=False)

agg_df=z.sort_values("PRICE", ascending=False)

agg_df.head()

###GÖREV-4:Index’te yer alan isimleri değişken ismine çeviriniz.

agg_df.reset_index()

agg_df = agg_df.reset_index()

agg_df.columns

###GÖREV-5:age değişkenini kategorik değişkene çeviriniz ve  agg_df’e ekleyiniz.

bins_= [0, 18, 23, 30, 40, 70]
labels_= ['0_18', '19_23', '24_30', '31_40', '41_70']

agg_df["AGE_CAT"]= pd.cut(agg_df["AGE"],bins=bins_, labels=labels_ )

agg_df.head(5)

###GÖREV-6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.##
#Yeni eklenecek değişkenin adı: customers_level_based
#Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.



agg_df["customer_level_based"]=[str(row[0].upper()) + "_" + str(row[1].upper()) + "_" + str(row[2].upper()) + "_" + str(row[5].upper()) for row in agg_df.values]

agg_df


agg_df=agg_df.groupby(["customer_level_based"]).agg({"PRICE":"mean"})

agg_df=agg_df.sort_values("PRICE",ascending=False).reset_index()

agg_df.head(20)

###GÖREV-7:Yeni müşterileri (personaları) segmentlere ayırınız.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D","C","B","A"])

agg_df.head(20)

agg_df.groupby(["SEGMENT"]).agg({"PRICE":["mean","max","sum"]})

agg_df.head(20)

agg_df[agg_df["SEGMENT"]=="C"].describe()

###GÖREV-8: Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz.

def new(agg_df, new_user):
    print(agg_df[agg_df["customer_level_based"]== new_user])

new(agg_df,"TUR_ANDROID_FEMALE_31_40")

new(agg_df,"FRA_IOS_FEMALE_31_40")
