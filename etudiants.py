#Importation de librairie
import streamlit as st
import pandas as pd
import joblib
#Importation de librairie
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk

# Charger le modèle et les encodeurs
with open('modele_entraine.pkl', 'rb') as fichier:
  modele=pk.load(fichier)
with open('label.pkl', 'rb') as fichier:
  label=pk.load(fichier)
  
#Chargement des données
df=pd.read_excel("MathE dataset.xlsx")


# Titre de l'application
st.title("Prédiction de Réponse d'un Étudiant par Pensezy")

#Menu de navigation 
menu=sl.sidebar.selectbox("Menu", ["Accueil", "Données", "Graphiques", "Prédiction"])

#Page d'accueil 
if menu=="Accueil": 
  sl.write("Notre outil est précieux pour l'analyse des performances éducatives en mathématiques, avec un focus sur les réponses des étudiants à des questions variées couvrant plusieurs domaines mathématiques.")
 
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
      

#Page des Prédictions
elif menu=="Prédiction":
    # Interface utilisateur
    st.write("Remplissez les informations suivantes pour prédire si l'étudiant répondra correctement à la question.")
    
    # Champs de saisie
    student_country = st.selectbox("Pays de l'étudiant", label["Student Country"].classes_)
    question_id = st.number_input("ID de la question", min_value=0, step=1)
    question_level = st.selectbox("Niveau de la question", label["Question Level"].classes_)
    topic = st.selectbox("Sujet", label["Topic"].classes_)
    subtopic = st.selectbox("Sous-sujet", label["Subtopic"].classes_)
    
    # Convertir les entrées utilisateur en valeurs numériques
    input_data = pd.DataFrame({
        "Student Country": [label["Student Country"].transform([student_country])[0]],
        "Question ID": [question_id],
        "Question Level": [label["Question Level"].transform([question_level])[0]],
        "Topic": [label["Topic"].transform([topic])[0]],
        "Subtopic": [label["Subtopic"].transform([subtopic])[0]],
    })
    
    # Bouton de prédiction
    if st.button("Prédire"):
        prediction = model.predict(input_data)[0]
        result = "Réponse correcte ✅" if prediction == 1 else "Réponse incorrecte ❌"
        st.subheader(f"Résultat : {result}")
