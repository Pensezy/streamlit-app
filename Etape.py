#importation de librairie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#chargement de la base de donnée
df=pd.read_excel('Iris.xlsx')

#Etapes 3. Preparer les données pour le modèle

#1. importation 
#from sklearn.model_selection import train_test_split
import sklearn.model_selection as sms

#séparer les caractéristiques et la cible
X=df.drop('Species', axis=1)
y = df['Species']

#2.
#Diviser les données en ensemble d'entraînement et de test 
X_train, X_test, y_train, y_test=sms.train_test_split (X, y, test_size=0.2,random_state=42)

#3
from sklearn.preprocessing import StandardScaler
#Normaliser les caractéristiques

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#Etapes 4. Créer et entraîner un modèle de classification (K-Nearest Neighbors)

#1.importation des bibliothèques
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sm
#from sklearn.metrics import confusion_matrix, accuracy_score
#from sklearn.metrics 

#Créer le modèle KNN
knn=KNeighborsClassifier(n_neighbors=5)#3

#2.
#Entrainons le modèle 
knn.fit(X_train, y_train)
