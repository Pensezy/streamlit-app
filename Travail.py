#importation de librairie
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#chargement de la base de donnée
#df=pd.read_excel('Iris.xlsx')
df=pd.read_csv('Iris.csv')


#Affichage des premières lignes du jeu de données
print(df.head())

#Statistiques descriptives 
print(df.describe())
