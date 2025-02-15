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


#Etape 6 question 2.
#Importation 
import sklearn.preprocessing as sp
import sklearn.linear_model as sl

#Normalisation Min-Max
scaler_minmax=sp.MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

#Normalisation Z-score
scaler_zscore=sp.StandardScaler()
X_train_zscore=scaler_zscore.fit_transform(X_train)
X_test_zscore=scaler_zscore.transform(X_test)

# Entraîner et évaluer le modèle avec les données normalisées
model = sl.LogisticRegression()

model.fit(X_train_minmax, y_train)
y_pred_minmax = model.predict(X_test_minmax)
accuracy_minmax = sm.accuracy_score(y_test, y_pred_minmax)

model.fit(X_train_zscore, y_train)
y_pred_zscore = model.predict(X_test_zscore)
accuracy_zscore = sm.accuracy_score(y_test, y_pred_zscore)

print(f"Précision avec normalisation Min-Max: {accuracy_minmax}")
print(f"Précision avec normalisation Z-score: {accuracy_zscore}")


#etape 7
#1 Optimisation des hypes-paramètres

#a.
# Normaliser les données (important pour KNN)
scaler = sp.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Définissons la grille d'hyperparamètres
param_grid = {
    'n_neighbors': range(1, 31),  # Tester de 1 à 30 voisins
    'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski'],  # Tester différentes distances
    'p': [1, 2] # Pour la distance de Minkowski (p=1 pour Manhattan, p=2 pour Euclidienne)
}

#Trouvons les meilleurs hyperparamètres
grid_search = sms.GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # cv pour la validation croisée, scoring pour la métrique d'évaluation
grid_search.fit(X_train_scaled, y_train)

# Afficher les meilleurs hyperparamètres et le score correspondant
print("Meilleurs hyperparamètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)

#On évalue le modèle 
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

# Afficher le rapport de classification et la matrice de confusion
print(sm.classification_report(y_test, y_pred))
print(sm.confusion_matrix(y_test, y_pred))

#b.
# Recherche aléatoire
param_distributions = {  # Utilisez param_distributions pour RandomizedSearchCV
    'n_neighbors': range(1, 31),
    'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski'],
    'p': [1, 2]
}

random_search = sms.RandomizedSearchCV(knn, param_distributions, n_iter=10, cv=5, scoring='accuracy')  # n_iter pour le nombre d'itérations
random_search.fit(X_train_scaled, y_train)

# Afficher les résultats
print("Meilleurs paramètres (recherche en grille):", grid_search.best_params_)
print("Meilleur score (recherche en grille):", grid_search.best_score_)

print("Meilleurs paramètres (recherche aléatoire):", random_search.best_params_)
print("Meilleur score (recherche aléatoire):", random_search.best_score_)
