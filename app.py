#Importation de librairie
import streamlit as sl
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk

#Chargement du modèle
with open('mon_modele.pkl', 'rb') as fichier:
  mon_modele=pickle.load(fichier)

#Titre de mon application 
sl.title("Pensezy Corporation")

#Menu de navigation 
menu=sl.sidebar.selectbox("Menu", ["Accueil", "Données", "Graphiques", "À propos"])

#Page d'accueil 
if menu=="Accueil": 
  sl.write("Cette application permet de visualiser et d'analyser des données. Les données utilisées ici sont célèbres dans le domaine des data sciences. Elles ont été collectées par Edgar Anderson [1]. Ce sont les mesures en centimètres des variables suivantes : longueur du sépale
(Sepal.Length), largeur du sépale (Sepal.Width), longueur du pétale (Petal.Length) et largeur du pétale
(Petal.Width) pour trois espèces d’iris : setosa, versicolor et virginica.")

#Page données
elif menu == "Données":
  sl.write("Voici les données : ")
  data=pd.DataFrame({"Nom": ["John", "Mary", "David"], "Age": [25, 31, 42]})
  #data=pd.read_excel("Iris.xlsx")
  sl.write(data)

#Page des graphiques
elif menu=="Graphiques":
  sl.write("Voici un graphque :")
  fig,ax=plt.subplots()
  ax.plot([1, 2, 3], [2, 4, 6])
  sl.pyplot(fig)

#Page à propos
elif menu =="À propos":
  sl.write("Cette application a été créée par Pensezy Corporation.")
  sl.write("Pour plus d'informations, veuillez nous contacter au numéro : +237 6 95 91 18 71")

#Bouton pour télécharger les données
#sl.sidebar.download_button("Télécharger les données", data.to_excel(index=false), "données.excel")

#Bouton pour afficher les crédits 
#sl.sidebar.bouton("Crédits", on_click=lambda:st.sidebar.write("Cette appliction utilise Streamlit et Pandas"))
