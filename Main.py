import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import math

print("---------------------------------------------------------------------------------------\n")

df = pd.read_csv('./Dataset.csv')

# Estraggo gli stati

states = df['Country'].values.tolist()

#Rimuovo i ripetuti e ordino

newStates = list(dict.fromkeys(states))
states=sorted(newStates)
print("Stati presenti:\n")
print("---------------------------------------------------------------------------------------")
print(states,sep = "\n")
print("---------------------------------------------------------------------------------------\n")

#Inserimento dello stato da predire

inputStates = input("Inserisci uno stato: ")
inputStates = inputStates.upper()
print("\n")

print("---------------------------------------------------------------------------------------\n")
#DF contenente tutti i dati provenienti dal datset relativi alla nazione inserita

specificStateDF = df[df.Country == inputStates]
print("Olimpiadi a cui ha partecipato e medaglie conquistate\n")

#Istogramma delle medaglie vinte fino al 2012

sns.catplot(x="Year", kind="count", data=specificStateDF,aspect=2); 
plt.show()

#Lista degli anni in cui la nazione inserita ha partecipato

years = df['Year'].values.tolist()
newYears = list(dict.fromkeys(years))

print("---------------------------------------------------------------------------------------\n")
#Schema medaglie

filterDf = specificStateDF.groupby(["Year", "Country"]).size().reset_index(name="Count")
filterDf['Cumulative_Sum'] = filterDf['Count'].cumsum()
print("Schema riassuntivo delle medaglie conquistate da {} \n".format(inputStates))
print(filterDf)
print("---------------------------------------------------------------------------------------\n")
tableAllCountry = df.groupby(["Year", "Country"]).size().reset_index(name="Count")
mat=[]
row=filterDf.values.tolist()
for r in row:
    year=r
    table=tableAllCountry[tableAllCountry.Year == year[0]]
    tableStateSpecifiedYear=table.values.tolist()
    a=[]
    for v in tableStateSpecifiedYear:
        a.append(v)
    mat.append(a)
i=0
quartiliState=[]
for m in mat:
    resultOlimp=sorted(m , key=lambda x: x[2])
    stateValues=filterDf.Count[i]
    i=i+1
    mat1=[]
    for r in resultOlimp:
        mat1.append(r[2])
    array=np.array(mat1)
    q1=np.quantile(array, 0.25)
    q2=np.quantile(array, 0.5)
    q3=np.quantile(array, 0.75)
    if stateValues < q1:
        quartiliState.append(1)
    elif stateValues < q2:
        quartiliState.append(2)
    else:
        quartiliState.append(3)
dfQuarts=pd.DataFrame(columns=['Year','Quartile'])
i=0
for y in filterDf.Year:
    dfQuart = pd.DataFrame({"Year":[y],"Quartile":[quartiliState[i]]})
    dfQuarts.loc[i]=[y,quartiliState[i]] 
    i=i+1
#Istogramma quartili
print("Grafico che mostra la posizione della nazione inserita rispetto ai quartili di ordine 0.25, 0.5, 0.75")
sns.catplot(x="Year",y="Quartile", kind="point", data=dfQuarts,aspect=2); 
plt.show()
print("1=la nazione si trova nel primo quartile rispetto a tutte le nazioni che hanno partecipato")
print("2=la nazione si trova nel secondo quartile rispetto a tutte le nazioni che hanno partecipato")
print("3=la nazione si trova nel terzo quartile rispetto a tutte le nazioni che hanno partecipato")
print("4=la nazione si trova nel quarto quartile rispetto a tutte le nazioni che hanno partecipato")
print("---------------------------------------------------------------------------------------\n")
#Preparazione per applicare la regressione

yearlist = filterDf['Year'].tolist()
countlist = filterDf['Count'].tolist()

#Calcolo Covarianza

covariance= np.cov(countlist,yearlist)[0][1]
print('- Covarianza: {}'.format(covariance))

print("\n---------------------------------------------------------------------------------------")

#Converto in array
yearlist=np.array(yearlist,dtype = int)
countlist=np.array(countlist,dtype = int)

#Preaprazione degli array
X = np.reshape(yearlist, (-1, 1))
Y = np.reshape(countlist, (-1, 1)) 

#Regressione lineare con apprendimento dal dataframe relativo alla nazione inserita
regsr=LinearRegression()
regsr.fit(X,Y)

#Inserimento anno a cui applicare la predizione
while True:
    try:
       inputYear = int(input('Inserisci un anno maggiore del 2012 , considerando che le olimpiadi si svolgono ogni 4 anni: '))
       if inputYear > 2012:
            break
       else:
            print ('Riprova , anno già presente')
    except ValueError:
        print ('Input non valido')

print("\n---------------------------------------------------------------------------------------\n")
#Definizione degli assi

to_predict_x= [inputYear]
to_predict_x= np.array(to_predict_x).reshape(-1,1)

predicted_y= regsr.predict(to_predict_x)

print("Totale medaglie Previste da  {} nel {}: {}\n".format(inputStates,inputYear,int(predicted_y))) 


#Coefficienti per la retta di regessione

m= regsr.coef_
print("Inclinazione (m): ",m)

c= regsr.intercept_
print("Intercetta (c): ",c)

#Visualizzazione grafico con retta di regressione

plt.title('Medaglie Grafico-Stato')  
plt.xlabel('Anni')  
plt.ylabel('Medaglie') 
plt.scatter(X,Y,color="green")

new_y=[ m*i+c for i in np.append(X,to_predict_x)]
new_y=np.array(new_y).reshape(-1,1)
plt.plot(np.append(X,to_predict_x),new_y,color="blue")
plt.show()

print("---------------------------------------------------------------------------------------\n")
#Calcolo indice di Pearson
corr, _ = pearsonr(countlist,yearlist)
print('\n- Indice Pearson: %.3f' % corr)



#Inserimento della predizione nel dataframe
lastSum=filterDf['Cumulative_Sum'].iloc[-1]
df2 = pd.DataFrame({"Year":[inputYear], 
                    "Country":[inputStates],
                    "Count":[int(predicted_y)],
                    "Cumulative_Sum":[int(predicted_y)+lastSum]
                    }) 
sortedDf = filterDf.append(df2,ignore_index = True)

#Calcolo differenza anni 

lastSum=sortedDf['Cumulative_Sum'].iloc[-1]
countYear = len(yearlist)
l= list(range(countYear))
base = l[-1]+2 
#+2 perché +1 è la differenza tra quelle che a cui ha partecipato e quella predetta e l'altro +1 è per il fatto che l'indice parte da 0

#MEDIA

avg = lastSum / base
print("\n- Media : {} \n".format(avg))

#VARIANZA

lst = sortedDf['Count'].tolist()
variance = np.var(lst)
print("\n- Varianza: {} \n".format(variance))

#Scarto quadratico medio
print("- Scarto quadratico medio (DEVIAZIONE STANDARD): {} \n".format(math.sqrt(variance)))

