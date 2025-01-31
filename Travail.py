#importation de librairie
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#chargement de la base de donnée
df=pd.read_csv('Iris.csv')

#Affichage des premières lignes du jeu de données
st.text(df.head())


# Titre de l'application
st.title("Interface utilisateur Pensezy")

# Introduction
st.write("Bienvenue dans cette application interactive ! Utilisez les éléments ci-dessous pour explorer les données.")

 # Affichage des données brutes (facultatif)
if st.checkbox("Afficher les données brutes"):
    st.write(df)

# Sélection des colonnes pour l'analyse
st.subheader("Sélection des colonnes")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
selected_cols = st.multiselect("Choisir les colonnes à analyser", numeric_cols)

if selected_cols:
    # Graphique interactif (exemple avec un histogramme)
    st.subheader("Visualisation des données")
    col_to_plot = st.selectbox("Choisir une colonne pour l'histogramme", selected_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col_to_plot], kde=True, ax=ax)
    st.pyplot(fig)

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df[selected_cols].describe())

    # Possibilité d'ajouter d'autres types de graphiques (barres, dispersion, etc.)
    # et d'autres analyses statistiques selon vos besoins.

else:
    st.write("Veuillez sélectionner au moins une colonne.")
