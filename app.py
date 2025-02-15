#Importation de librairie
import streamlit as sl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.linear_model import LogisticRegression

#Chargement du modèle 
with open('modele_iris.pkl', 'rb') as fichier:
  modele=pk.load(fichier)
  
#Chargement des données
df=pd.read_excel("Iris.xlsx")

#Titre de mon application 
sl.title("Pensezy Corporation")

#Menu de navigation 
menu=sl.sidebar.selectbox("Menu", ["Accueil", "Données", "Graphiques", "Prédictions"])

#Page d'accueil 
if menu=="Accueil": 
  sl.write("Cette application permet de visualiser et d'analyser des données. Les données utilisées ici sont célèbres dans le domaine des data sciences. Elles ont été collectées par Edgar Anderson [1]. Ce sont les mesures en centimètres des variables suivantes : longueur du sépale (Sepal.Length), largeur du sépale (Sepal.Width), longueur du pétale (Petal.Length) et largeur du pétale (Petal.Width) pour trois espèces d’iris : setosa, versicolor et virginica.")
 
#Page données
elif menu == "Données":
  sl.write("Voici les données : ")
  sl.write(df)

#Page des graphiques
elif menu=="Graphiques":
  sl.subheader("Sélection des colonnes")
  #Listons les colonnes
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  selected_cols = sl.multiselect("Choisir les colonnes à analyser", numeric_cols)
  
  if selected_cols:
      # Graphique interactif (exemple avec un histogramme)
      sl.subheader("Visualisation des données")
      col_to_plot = sl.selectbox("Choisir une colonne pour l'histogramme", selected_cols)
      fig, ax = plt.subplots()
      sns.histplot(df[col_to_plot], kde=True, ax=ax)
      sl.pyplot(fig)
      # Statistiques descriptives
      sl.subheader("Statistiques descriptives")
      sl.write(df[selected_cols].describe())

#Page de prédictions
  col1, col2 = sl.columns(2)
  feature1 = col1.selectbox("Feature 1", df.columns)
  feature2 = col2.selectbox("Feature 2", df.columns)
  if sl.button("Faire des prédictions"):
    predictions = modele.predict(df[[feature1, feature2]])

    # Affichage des résultats
    sl.write("Prédictions :")
    sl.write(predictions)

    # Graphique
    fig, ax = plt.subplots()
    ax.scatter(df[feature1], df[feature2], c=predictions)
    sl.pyplot(fig)
