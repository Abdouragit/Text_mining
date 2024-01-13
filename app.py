import streamlit as st
import sqlite3
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim

# Fonction pour charger les données depuis la base de données
#@st.cache_resource
#def charger_donnees():
    #conn = sqlite3.connect('Database.db')
    #query = 'SELECT * FROM d_ville'
    #df = pd.read_sql(query, conn)
    #conn.close()
    #return df


def charger_donnees():
    df = pd.read_csv('Jobs_V4.csv')
    return df

# Chargement des données
df = charger_donnees()

st.set_page_config(layout="wide")

# Liste des onglets
onglets = ["Accueil", "Statistiques", "Cartographie", "Analyse du corpus"]

# Première "fenêtre"
with st.sidebar:
    
    st.write("##")
    st.write("##")
    st.image("logo/job.png")
    st.title("Sommaire")

    # Utiliser le cache pour les onglets
    onglet_selectionne = st.sidebar.radio("Sélectionnez un onglet ", onglets, format_func=lambda x:x)

# Deuxième "fenêtre"
with st.container():
    st.title(f"{onglet_selectionne}")

    # Afficher le tableau de bord sélectionné
    if onglet_selectionne == "Analyse du corpus":
        st.subheader("Données depuis la base de données :")
        st.dataframe(df[['intitule', 'poste', 'profil']].head(5))
        
        data_scientist_df= df.loc[df['poste'] == 'data scientist']
        data_engineer_df= df.loc[df['poste'] == 'data engineer']
        data_scientist_df['ID'] = range(1, len(data_scientist_df) + 1)
        data_scientist_df = data_scientist_df[['ID'] + [col for col in data_scientist_df.columns if col != 'ID']]
        data_engineer_df['ID'] = range(1, len(data_engineer_df) + 1)
        data_engineer_df = data_engineer_df[['ID'] + [col for col in data_engineer_df.columns if col != 'ID']]
        DS= data_scientist_df.copy()
        DE =data_engineer_df.copy()
        DS['profil'] = DS['profil'].astype('object')
        DE['profil'] = DE['profil'].astype('object')
        corpusDS = [str(doc).lower() if isinstance(doc, (str, float)) else '' for doc in DS['profil'].tolist()]
        corpusDE = [str(doc).lower() if isinstance(doc, (str, float)) else '' for doc in DE['profil'].tolist()]
        chiffres = list("0123456789")
        corpusDS= ["".join([mot for mot in list(doc) if not mot in chiffres]) for doc in corpusDS]
        corpusDE= ["".join([mot for mot in list(doc) if not mot in chiffres]) for doc in corpusDE]
        corpusDS = [doc.lower() for doc in corpusDS]
        corpusDE = [doc.lower() for doc in corpusDE]
        ponctuations = list(string.punctuation)
        corpusDS = ["".join([char for char in list(doc) if not (char in ponctuations)]) for doc in corpusDS]
        corpusDE = ["".join([char for char in list(doc) if not (char in ponctuations)]) for doc in corpusDE]
        corpusDS = [s.replace("\n","") for s in corpusDS]
        corpusDE = [s.replace("\n","") for s in corpusDE]
        corpus_tkDS = [word_tokenize(doc) for doc in corpusDS]
        corpus_tkDE = [word_tokenize(doc) for doc in corpusDE]
        lem = WordNetLemmatizer()
        corpus_lmDS = [[lem.lemmatize(mot) for mot in doc] for doc in corpus_tkDS]
        corpus_lmDE = [[lem.lemmatize(mot) for mot in doc] for doc in corpus_tkDE]
        mots_vides = stopwords.words('french')
        corpus_swDS = [[mot for mot in doc if not (mot in mots_vides)] for doc in corpus_lmDS]
        corpus_swDE = [[mot for mot in doc if not (mot in mots_vides)] for doc in corpus_lmDE]
        corpus_swDS = [[mot for mot in doc if len(mot) >= 3] for doc in corpus_swDS]
        corpus_swDE = [[mot for mot in doc if len(mot) >= 3] for doc in corpus_swDE]
        
        # Créer le graphique Word2Vec
        modeleDS = Word2Vec(corpus_swDS, vector_size=5, window=3)
        modeleDE = Word2Vec(corpus_swDE, vector_size=5, window=3)
        wordsDS = modeleDS.wv
        wordsDE = modeleDE.wv
        terms_data_scientist = list(wordsDS.index_to_key)
        terms_data_engineer = list(wordsDE.index_to_key)
        
        nombre_mots_a_afficher = st.slider("Veuillez choisir le nombre de mots à afficher", 1, 20, 10)
        st.markdown("<h3 style='text-align: center;'>Représentation des vecteurs associés aux termes les plus fréquents des profils Data Scientist et Data Engineer</h3>", unsafe_allow_html=True)
        top_terms_data_scientist = terms_data_scientist[:nombre_mots_a_afficher]
        top_terms_data_engineer = terms_data_engineer[:nombre_mots_a_afficher]
        vectors_data_scientist = wordsDS[top_terms_data_scientist]
        vectors_data_engineer = wordsDE[top_terms_data_engineer]
        df_scientist = pd.DataFrame(vectors_data_scientist, columns=[f'V{i}' for i in range(vectors_data_scientist.shape[1])], index=top_terms_data_scientist)
        df_engineer = pd.DataFrame(vectors_data_engineer, columns=[f'V{i}' for i in range(vectors_data_engineer.shape[1])], index=top_terms_data_engineer)

        # Créer une figure matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Scatter plot pour Data Scientist
        ax1.scatter(df_scientist['V0'], df_scientist['V1'], s=50, label='Data Scientist')
        for i in range(df_scientist.shape[0]):
            ax1.annotate(df_scientist.index[i], (df_scientist['V0'][i], df_scientist['V1'][i]))

        # Scatter plot pour Data Engineer
        ax2.scatter(df_engineer['V0'], df_engineer['V1'], s=50, label='Data Engineer', marker='x')
        for i in range(df_engineer.shape[0]):
            ax2.annotate(df_engineer.index[i], (df_engineer['V0'][i], df_engineer['V1'][i]))

        # Afficher les légendes et les titres
        ax1.legend()
        ax2.legend()

        # Afficher la figure dans Streamlit
        st.pyplot(fig)      

    elif onglet_selectionne == "Dashboard 2":
        st.subheader("Dashboard 2 :")
        # Ajoutez ici le contenu spécifique au Dashboard 2

