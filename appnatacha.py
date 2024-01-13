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

def preprocess_text_column(column):
    # Supprimer les valeurs NaN de la colonne
    column = column.dropna()
    # Transformer la colonne en liste
    corpus = column.tolist()
    # Passer en minuscule
    corpus = [doc.lower() for doc in corpus]
    # Retirer les chiffres dans l'ensemble du corpus
    chiffres = list("0123456789")
    corpus = ["".join([mot for mot in list(doc) if not mot in chiffres]) for doc in corpus]
    # Retrait des ponctuations
    ponctuations = set(string.punctuation)
    corpus = ["".join([char for char in list(doc) if not (char in ponctuations)]) for doc in corpus]
    # Enlever les "\n"
    corpus = [s.replace("\n", "") for s in corpus]
    # Transformer le corpus en liste de listes (les documents) par tokenisation
    corpus_tk = [word_tokenize(doc) for doc in corpus]
    # Lemmatisation
    lem = WordNetLemmatizer()
    corpus_lm = [[lem.lemmatize(mot) for mot in doc] for doc in corpus_tk]
    # Charger les stopwords
    mots_vides = stopwords.words('french')
    # Suppression des stopwords
    corpus_sw = [[mot for mot in doc if not (mot in mots_vides)] for doc in corpus_lm]
    # Retirer les tokens de moins de 3 lettres
    corpus_sw = [[mot for mot in doc if len(mot) >= 3] for doc in corpus_sw]
    return corpus_sw

def create_word2vec_plot(words, top_terms, label, marker='o'):
    vectors = words[top_terms]
    df = pd.DataFrame(vectors, columns=[f'V{i}' for i in range(vectors.shape[1])], index=top_terms)
    
    # Créer une figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(df['V0'], df['V1'], s=50, label=label, marker=marker)
    for i in range(df.shape[0]):
        ax.annotate(df.index[i], (df['V0'][i], df['V1'][i]))

    # Afficher les légendes et les titres
    ax.legend()

    # Retourner la figure
    return fig

def create_word2vec_graph(corpus, vector_size=2, window=5, marker='o', label=None, nombre_mots_a_afficher=None):
    modele = Word2Vec(corpus, vector_size=vector_size, window=window)
    words = modele.wv
    terms = list(words.index_to_key)
    top_terms = terms[:nombre_mots_a_afficher] if nombre_mots_a_afficher is not None else terms
    return create_word2vec_plot(words, top_terms, label, marker)


def charger_donnees():
    conn = sqlite3.connect('Database.db')
    query = '''SELECT 
        offres.intitule,
        offres.poste,
        offres.profil,
        offres.description_offre,
        offres.date_creation,
        offres.salaire_min_annuel,
        offres.salaire_max_annuel,
        offres.salaire_annuel_mean,
        offres.qualification_libelle,
        offres.experience,
        offres.type_contrat,
        offres.secteur_activite,
        offres.id_ville,
        offres.id_entreprise,
        d_ville.id,
        d_ville.ville,
        d_ville.latitude,
        d_ville.longitude,
        d_ville.code_postal,
        d_ville.departement,
        d_entreprise.id,
        d_entreprise.entreprise_nom,
        d_entreprise.entreprise_description,
        h_departement.departement,
        h_departement.departement_nom,
        h_departement.region
        
        FROM offres
        INNER JOIN d_ville ON offres.id_ville = d_ville.id
        INNER JOIN d_entreprise ON offres.id_entreprise = d_entreprise.id
        INNER JOIN h_departement ON d_ville.departement = h_departement.departement;
        '''
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Chargement des données
df = charger_donnees()

#dataframe pour data-engineer et data-scientist
data_engineer_df = df[df['poste']== 'data engineer']
data_scientist_df = df[df['poste']== 'data scientist']

# Définir la configuration de la page
st.set_page_config(layout="wide",  # Changer le titre de la page,
)

# Liste des onglets
onglets = ["Accueil", "Statistiques", "Cartographie", "Analyse du corpus"]

# Première "fenêtre"
with st.sidebar:
    st.write("##")
    st.write("##")
    st.image("logo/job.png", width=140)
    st.title("Sommaire")
    # Utiliser le cache pour les onglets
    onglet_selectionne = st.sidebar.radio("Sélectionnez un onglet ", onglets, format_func=lambda x:x)

# Deuxième "fenêtre"
with st.container():
 
    if onglet_selectionne == "Accueil":
        st.markdown('<p class="first_titre" style="text-align: center; font-size: 50px; font-weight: bold;">Projet Text Mining</p>', unsafe_allow_html=True)
        st.write("---")
        c1, c2, c3 = st.columns((4, 0.5, 0.5))
        #Colonne de droite pour l'image
        with c2:
            st.write("##")
            st.write("##")
            st.write("##")
            st.image("logo/python.png", width=90)
            st.write("##")
            st.image("logo/streamlit.png", width=90)
            st.write("##")
        
        with c3:
            st.write("##")
            st.write("##")
            st.write("##")
            st.image("logo/poleemploi.png", width=110)
            st.image("logo/apec.png", width=110)
            st.write("##")
        with c1:
            st.write("##")
            st.markdown("## Bienvenue sur notre application Streamlit !")
            st.markdown(
                """
                <p style="font-size: 20px; text-align: justify;">
                Ce projet s'inscrit dans le cours de Text Mining du Master 2 SISE. Cette application est connectée à une base de données Sqlite contenant des données sur des offres d'emplois.
                Ces offres ont été extraites des sites de Pôle Emploi et de l'Apec, par web scraping.
                
                <p style="font-size: 20px; text-align: justify;">
                Cette application intéractive sert de support pour l'exploration et l'analyse de ces offres en se focalisant sur des postes de Data scientist et Data Engineer, répartis à travers la France métropolitaine.
                Cette application renvoie des représentations cartographiques interactives et propose notamment des analyses régionales et départementales des offres d'emplois.
                
                <p style="font-size: 20px; text-align: justify;">
                Nous vous souhaitons une bonne naviguation!
                <p style="font-size: 20px; text-align: justify;">
                Auteurs: Martin Revel, Abdourahmane Ndiaye, Natacha Perez
                
                """,
                unsafe_allow_html=True
            )

    # Afficher le tableau de bord sélectionné
    if onglet_selectionne == "Analyse du corpus":
        st.title(f"{onglet_selectionne}")
        st.subheader("Données depuis la base de données :")
        st.dataframe(df[['intitule', 'poste', 'profil']].head(5))
        
        corpus_data_scientist = preprocess_text_column(data_scientist_df['profil'])
        corpus_data_engineer = preprocess_text_column(data_engineer_df['profil'])
        
        st.subheader("Objectif: comparaison des différences et similitudes entre les profils recherchés chez les Data Scientist et Data Engineer\n")
        nombre_mots_a_afficher = st.slider("Veuillez choisir le nombre de mots à afficher", 1, 20, 10)
        st.markdown("<h3 style='text-align: center; color: orange;'>Représentation des vecteurs associés aux termes les plus fréquents des profils Data Scientist et Data Engineer</h3>", unsafe_allow_html=True)
    
        fig_ds = create_word2vec_graph(corpus_data_scientist, label='Data Scientist', nombre_mots_a_afficher=nombre_mots_a_afficher)
        fig_de = create_word2vec_graph(corpus_data_engineer, label='Data Engineer', marker='x', nombre_mots_a_afficher=nombre_mots_a_afficher)

        # Afficher les deux graphiques côte à côte
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(fig_ds)

        with col2:
            st.pyplot(fig_de)
        
        interpretation_text = """
        **Interprétation :**\n
        Chaque point sur le graphique représente un terme, et sa position est déterminée par les valeurs des deux dimensions du vecteur associé à ce terme.
        La distance entre les points sur le graphique reflète la similarité entre les termes. Des points proches indiquent des termes similaires en termes de contexte ou de co-occurrence dans les corpus.
        À l'inverse, des termes éloignés indiquent une différence dans la manière dont ils sont utilisés.
        """
        # Afficher le texte d'interprétation
        st.markdown(interpretation_text, unsafe_allow_html=True)

    elif onglet_selectionne == "Statistiques":
        st.subheader("Dashboard 2 :")
        # Ajoutez ici le contenu spécifique au Dashboard 2
