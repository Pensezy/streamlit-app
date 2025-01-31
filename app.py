#%%streamlit run

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Charger les données
data = pd.read_excel("Iris.xlsx")

print("Bonjour pensezy")
"""
# Charger le modèle
model = LinearRegression()
model.fit(data[['feature1', 'feature2']], data['target'])

st.title("Déploiement de modèle avec Streamlit")

feature1 = st.slider("Feature 1", min_value=0, max_value=100)
feature2 = st.slider("Feature 2", min_value=0, max_value=100)

if st.button("Prédire"):
    prediction = model.predict([[feature1, feature2]])
    st.write(f"Prédiction : {prediction[0]}")
"""
