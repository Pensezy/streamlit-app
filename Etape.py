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


#Etape 5: Evaluer le modèle

#1.
#Prédire les classes de l'ensemble de test
y_pred=knn.predict(X_test)

# 2.Affichons la matrice de confusion 
conf_matrix = sm.confusion_matrix (y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique())
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.show()

#3.
#Calculer l'exactitude
accuracy = sm.accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle: {accuracy*100:.2f}%")

#Afficher le rapport de classification 
print("Rapport de classification : \n", sm.classification_report(y_test, y_pred))


#Etape 6
#1. Analysons les résultats du modèle 
precision=sm.precision_score(y_test, y_pred, average='weighted')
rappel=sm.recall_score(y_test, y_pred, average='weighted')
f1=sm.f1_score(y_test,y_pred,average='weighted')
#Calculons les positifs
TN=conf_matrix[0,0] #Les éléments de la diagonale principale : représentent les prédictions correctes
FP= conf_matrix[0,1] #Les éléments au dessus de la diagonale principale : représentent les prédictions incorrect (faux positifs)
FN=conf_matrix[1,0]#Les éléments en dessous de la diagonale principale : représentent les prédictions incorrectes (Faux négatifs)
TP=conf_matrix[1,1]#Les éléments en dehors de la diagonale principale : représentent les prédictions incorrect (Vrai négatifs)

erreur=(FP + FN)/(TP+FP + FN + TN)
print("Précision :", precision)
print("Rappel :", rappel)
print("Valeur F1 :", f1)
print("Erreur :", erreur)

plt.imshow(conf_matrix,interpolation='nearest')
plt.title("Matrice de confusion")
plt.colorbar()
plt.show()
