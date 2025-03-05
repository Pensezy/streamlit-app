import streamlit as st
import pandas as pd
import joblib

# Charger le modèle et les encodeurs
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Titre de l'application
st.title("Prédiction de Réponse d'un Étudiant")

# Interface utilisateur
st.write("Remplissez les informations suivantes pour prédire si l'étudiant répondra correctement à la question.")

# Champs de saisie
student_country = st.selectbox("Pays de l'étudiant", label_encoders["Student Country"].classes_)
question_id = st.number_input("ID de la question", min_value=0, step=1)
question_level = st.selectbox("Niveau de la question", label_encoders["Question Level"].classes_)
topic = st.selectbox("Sujet", label_encoders["Topic"].classes_)
subtopic = st.selectbox("Sous-sujet", label_encoders["Subtopic"].classes_)

# Convertir les entrées utilisateur en valeurs numériques
input_data = pd.DataFrame({
    "Student Country": [label_encoders["Student Country"].transform([student_country])[0]],
    "Question ID": [question_id],
    "Question Level": [label_encoders["Question Level"].transform([question_level])[0]],
    "Topic": [label_encoders["Topic"].transform([topic])[0]],
    "Subtopic": [label_encoders["Subtopic"].transform([subtopic])[0]],
})

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)[0]
    result = "Réponse correcte ✅" if prediction == 1 else "Réponse incorrecte ❌"
    st.subheader(f"Résultat : {result}")
