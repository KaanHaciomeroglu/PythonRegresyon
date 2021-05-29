# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:03:08 2021

@author: KAAN
"""


import seaborn as snd
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
import openpyxl

veriSeti = pd.read_excel('istanbul_veri.xlsx')
#İndex Olarak Ayları Atama
veriSeti.index= ["Ocak", "Şubat", "Mart", "Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık" ]
del veriSeti["Ay"]


veriSeti2 = pd.read_excel('veri2.xlsx')    #yıllara göre nüfus miktarı ve toplam su 
veriSeti2["Yıllar"] = pd.Series(["2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"],index=veriSeti2.index)
veriSeti2 = veriSeti2.rename(columns={"Su Miktarı": "SuMiktarı"})
veriSeti2 = veriSeti2.astype(int)

#Özet Bilgileri Verir
veriSeti.describe()
veriSeti2.describe()

# Sutunlardaki eksik verinin kac adet oldugunun bulunmasi
print(veriSeti.isnull().sum())
print(veriSeti2.isnull().sum())


#Normalize
n_veriSeti = (veriSeti - np.min(veriSeti))/(np.max(veriSeti)-np.min(veriSeti))


#Mevsimlere Ayırma
ilkbahar = veriSeti[2:5]
yaz = veriSeti[5:8]
sonbahar = veriSeti[8:11]
kis=veriSeti.iloc[[11,0,1],]


#Mevsim Ortalamaları Bulma
ilkbaharort = ilkbahar.mean()
round(ilkbaharort.mean(),2)
ilkbaharort = ilkbaharort.to_frame()
ilkbaharort.rename(columns={0: 'İlk Bahar'}, inplace=True)

yazort = yaz.mean()
round(yazort.mean(),2)
yazort = yazort.to_frame()
yazort.rename(columns={0: 'Yaz'}, inplace=True)

sonbaharort = sonbahar.mean()
round(sonbaharort.mean(),2)
sonbaharort = sonbaharort.to_frame()
sonbaharort.rename(columns={0: 'Son Bahar'}, inplace=True)

kisort = kis.mean()
round(kisort.mean(),2)
kisort = kisort.to_frame()
kisort.rename(columns={0: 'Kış'}, inplace=True)

#Mevsim Ortalamalarını Birleştirme
mevsimler = pd.concat([ilkbaharort,yazort,sonbaharort,kisort], join = "outer", axis=1)


mevsim_normalize = (mevsimler - np.min(mevsimler))/(np.max(mevsimler)-np.min(mevsimler))

#Aylara Göre Verilen Temiz Su Ortalaması (PlotBox)
mevsimler.plot.box(grid='True', color="red")
plt.title("Aylara Göre Verilen Temiz Su Ortalaması")
plt.show()

#Çizgi grafiği
plt.plot (ilkbahar, "o:r")
plt.title("İlk Bahar")

plt.plot (yaz, "o:r")
plt.title("Yaz")

plt.plot (sonbahar, "o:r")
plt.title("Son Bahar")

plt.plot (kis, "o:r")
plt.title("Kış")



#Isı haritası
import seaborn as sns
corr = n_veriSeti.corr()
sns.heatmap(
    corr, 
    annot = True, # Korelasyon degerlerinin grafigin uzerine yazdirma
    square=True, # Kutularin kare bicimde gosterilmesi
    cmap="Oranges" # Renklendirme secenegi
)




################################################
############# BASİT REGRESYON ################## 
################################################


X= veriSeti2[["Yıllar"]]    #BAĞIMSIZ DEĞİŞKEN
y= veriSeti2[["SuMiktarı"]]   #BAĞIMLI DEĞİŞKEN


#Regresyon Modelini Eğitme
from sklearn.linear_model import LinearRegression
reg =LinearRegression()

#fit metoduyla doğrusal bir fit elde ediyoruz 
model=reg.fit(X,y)
model

#B0 katsayısını Çağırmak için
model.intercept_
#B1 katsayısını çağırmak için
model.coef_

#R2 değeri
model.score(X,y) ##HATALI
#Bağımsız değişkenleini kullandığımızda bağımlı değişkenimizdeki değişimin yüzdekaçını açıkladığımızı söyler.



import seaborn as sns
g= sns.regplot(veriSeti2["Yıllar"], veriSeti2["SuMiktarı"],ci=None,scatter_kws={'color':'r','s':9})
g.set_title("Yıla göre verilen toplam su Tahmin Regresyon Model")
g.set_ylabel("Su Miktarı")
g.set_xlabel("Yıllar")


#TAHMİN YAPMA
model.intercept_+model.coef_*2014
model.predict([[2014]])

yeni_veri=[[0.80],[0.95],[1.10]]
model.predict(yeni_veri)

#HATALAR VE HATA KARELER ORTALAMASI
y.head()  #GERÇEK VERİ

#Kurmuş olduğumuz modeli kullanarak bir tahmin edilen çağırıyoruz 
model.predict(X)[0:6]
gercek_y=y[0:11]
tahmin_edilen_y=pd.DataFrame(model.predict(X)[0:11])

hatalar=pd.concat([gercek_y,tahmin_edilen_y],axis=1)
hatalar

hatalar.columns=["gercek_y","tahmin_edilen_y"]
hatalar

#Gerçek değerlerden tahmin değerlerini çıkartıp hatayı buluyoruz
hatalar["hata"]=hatalar["gercek_y"]-hatalar["tahmin_edilen_y"]
hatalar

#pozitif ve negatif değerlerinin birbirini götürmesini engellemek için karesini alıyoruz
hatalar["hata_kareler"]=hatalar["hata"]**2
hatalar

np.mean(hatalar["hata_kareler"])

##Sonradan Eklenenler
import statsmodels.api as sm
# Sabitin eklenmesi
x = sm.add_constant(X)

# Modelin çalıştırılması
model = sm.OLS(y,X).fit()

# Modelin yorumlanacağı tablo
print(model.summary())


################################################

#Derste eklediğimiz kısım
#import statsmodels.api as sm
#x = veriSeti2[["Yıllar"]] # bağımsız değişken (independent variable)
#y_gercek = veriSeti2[["SuMiktarı"]] # bağımlı değişken (target, dependent variable)
# Sabitin eklenmesi
#x = sm.add_constant(X)
# Modelin çalıştırılması
#model = sm.OLS(y,X).fit()
# Modelin yorumlanacağı tablo
#print(model.summary())


################################################
############# Polinomsal REGRESYON #############
################################################


from sklearn.preprocessing import PolynomialFeatures
#polinomsal grafiğimiz için derece alıyoruz
poly_regressor = PolynomialFeatures(degree = 20)  #degree değiştirilerek düşük öğrenme veya yüksek öğrenme uygulanabilir

X_poly = poly_regressor.fit_transform(X)
#Regresyon modelini eğitme
pol_regressor = LinearRegression()
pol_regressor.fit(X_poly, y )

#polinomsal regresyon grafiği
def polynomialRegressionVisual():
    plt.scatter(X, y, color = 'red')
    plt.plot(X, pol_regressor.predict(poly_regressor.fit_transform(X)), color='blue')
    plt.title("Polinomsal Regresyon Sonucu")
    plt.xlabel('Yıllar')
    plt.ylabel('Su Miktarı')
    plt.grid(True)
    plt.show()
    return
polynomialRegressionVisual()


################################################
############# Çoklu REGRESYON ################## 
################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



#bağımlı ve bağısız değişken oluşturma
yy = veriSeti2[["Yıllar"]]    #BAĞIMLI değişken
XX = veriSeti2.iloc[:,[0,1]]  #BAĞIMSIZ DEĞİŞKEN  #Nüfus / Su Miktarı

#veriyi eğitim ve test seti olmak üzere ayırdık %20 lik bir boyut aldık test setleri için
from sklearn.model_selection import train_test_split
XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size = 0.2, random_state = 0)

#Regresyon modelini eğitme
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XX_train, yy_train)

#bilgisayara takhin yaptırdık
yy_pred = regressor.predict(XX_test)

#Geriye Doğru Eleme Yöntemi ile Modeli Kurmak
XX = np.append(arr = np.ones((11,1)).astype(int), values = XX, axis = 1) 

#BİRİNCİ TUR
import statsmodels.api as sm
XX_opt = XX[:, [0,1,2]]
regressor_OLS = sm.OLS(endog=yy, exog=XX_opt).fit()
regressor_OLS.summary()

#İKİNCİ TUR
XX_opt = XX[:, [0,2]]
regressor_OLS = sm.OLS(endog=yy, exog=XX_opt).fit()
regressor_OLS.summary()
# Sabit değişken Ve Nüfus 
# P Değeri = Anlamlılık Değeri(0.05)

#	https://www.veribilimiokulu.com/coklu-regresyon-multiple-regression-python-ile-uygulama-2/
#	https://www.veribilimiokulu.com/r-ile-coklu-dogrusal-regresyonbaglanim-cozumlemesi/
#	https://www.veribilimiokulu.com/coklu-regresyon-multiple-regression-python-ile-uygulama-1/	
#	https://www.nufusu.com/il/istanbul-nufusu
#	https://data.ibb.gov.tr/dataset/istanbul-a-verilen-temiz-su-miktarlari/resource/27bdb043-0051-49df-bd7c-b68f60f31247
#	https://slideplayer.biz.tr/slide/12150569/
#	https://tr.sunflowercreations.org/449090-how-are-iloc-and-loc-YULMON
#	https://sedatsen.files.wordpress.com/2016/11/8-sunum.pdf
#	https://yigitsener.medium.com/makine-%C3%B6%C4%9Frenmesinde-python-ile-basit-do%C4%9Frusal-regresyon-modelinin-kurulmas%C4%B1-ve-yorumlanmas%C4%B1-4cf918e1adf
#	https://yigitsener.medium.com/polinomsal-polynomial-regresyon-ve-python-uygulamas%C4%B1-f742fb61a158 
