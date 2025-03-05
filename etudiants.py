import streamlit as st
import pandas as pd
import joblib

# Charger le modèle et les encodeurs
model = joblib.load("modele_entraine.pkl")
label = joblib.load("label.pkl")

# Titre de l'application
st.title("Prédiction de Réponse d'un Étudiant par Pensezy")

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
