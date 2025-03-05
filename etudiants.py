import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('modele_entraine.pkl')

# Titre de l'application
st.title("Mon Application de Prédiction des performances")

# Formulaire pour saisir les données
st.header("Entrez les données pour la prédiction")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Bouton pour faire la prédiction
if st.button("Prédire"):
    # Préparer les données pour la prédiction
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Faire la prédiction
    prediction = model.predict(input_data)
    
    # Afficher le résultat
    st.success(f"La prédiction est : {prediction[0]}")
