import streamlit as st
import sqlite3
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import string
ponctuations = list(string.punctuation)
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('omw-1.4')
import gensim
import re
from collections import Counter
import seaborn as sns



st.set_page_config(layout="wide")

# histogramme
def plot_most_common_words(corpus, title, ax, max_words=10):
    flat_list = [word for sublist in corpus for word in sublist]
    word_freq = Counter(flat_list)
    most_common = dict(word_freq.most_common(max_words))
    ax.bar(most_common.keys(), most_common.values())
    ax.set_title(title)
    ax.set_xlabel('Mots')
    ax.set_ylabel('Fréquence')

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

# Fonction pour charger les données depuis la base de données
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

# fonction pour créer un box plot des salaires
def plot_salary_boxplot(data, title, ax):
    sns.boxplot(x='poste', y='salaire_annuel_mean', data=data, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Poste')
    ax.set_ylabel('Salaire Annuel (en euros)')

# Chargement des données
df = charger_donnees()

# Liste des onglets
onglets = ["Accueil", "Statistiques générales", "Cartographie", "Analyse du corpus"]

# Première "fenêtre"
with st.sidebar:
    
    st.write("##")
    st.write("##")
    #st.image("logo/job.png")
    st.title("Sommaire")

    # Utiliser le cache pour les onglets
    onglet_selectionne = st.sidebar.radio("Sélectionnez un onglet ", onglets, format_func=lambda x:x)

# Deuxième "fenêtre"
with st.container():
    st.title(f"{onglet_selectionne}")

    # Afficher le tableau de bord sélectionné
    if onglet_selectionne == "Statistiques générales":

        #dataframe pour data-engineer
        data_engineer_df = df[df['poste']== 'data engineer']

        #dataframe pour data-scientist
        data_scientist_df = df[df['poste']== 'data scientist']

        # Filtrer les valeurs None
        data_engineer_filtered_df = data_engineer_df.dropna(subset=['salaire_annuel_mean'])
        data_scientist_filtered_df = data_scientist_df.dropna(subset=['salaire_annuel_mean'])

        #transformer salaire_annuel_mean en type int
        data_engineer_filtered_df['salaire_annuel_mean'] = data_engineer_filtered_df['salaire_annuel_mean'].astype(float)
        data_scientist_filtered_df['salaire_annuel_mean'] = data_scientist_filtered_df['salaire_annuel_mean'].astype(float)

        #liste des régions
        regions = df['region'].unique().tolist()

        #liste des départements
        departements = df['departement_nom'].unique().tolist()

        liste_options = ['France', 'Région', 'Département']
        # Affichage du menu déroulant
        granularité_selectionnee = st.selectbox('Sélectionnez la granularité', liste_options)

        # Ajouter un curseur pour sélectionner le nombre de mots dans le WordCloud
        nombre_mots = st.slider("Sélectionnez le nombre de mots dans le WordCloud et l'Histogramme", min_value=10, max_value=50, value=25)

        if granularité_selectionnee == "France":

            # Affichage du nombre total d'offres dans une boîte d'information
            st.info(f"**Nombre total d'offres : {len(df)}**")

            # Diviser la page en deux colonnes
            col1, col2 = st.columns(2)
            
            # Afficher le Wordcloud data Engineer dans la première colonne
            with col1:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres Data Engineer : {len(data_engineer_df)}**")

                st.subheader("Wordcloud data Engineer :")
                #corpus nettoyé de data ingé
                corpus_data_engineer = preprocess_text_column(data_engineer_df['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_engineer])
                # Créer l'objet WordCloud
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                # Utiliser BytesIO pour enregistrer l'image
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')  # Masquer les axes
                plt.savefig(img_buffer, format='png')
                # Afficher l'image avec st.image()
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme Data Engineer:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_engineer, "Mots les plus fréquents - Data Engineer", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot salaire data Engineer :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(data_engineer_filtered_df, "Box Plot des Salaires - Data Engineer", ax)
                st.pyplot(fig)        

            # Afficher le Wordcloud data Scientist dans la deuxième colonne
            with col2:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres Data Scientist : {len(data_scientist_df)}**")

                st.subheader("Wordcloud data Scientist :")

                #corpus nettoyé de data scientist
                corpus_data_scientist = preprocess_text_column(data_scientist_df['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_scientist])
                # Créer l'objet WordCloud
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                # Utiliser BytesIO pour enregistrer l'image
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')  # Masquer les axes
                plt.savefig(img_buffer, format='png')
                # Afficher l'image avec st.image()
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme Data Scientist:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_scientist, "Mots les plus fréquents - Data Scientist", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot salaire data Scientist :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(data_scientist_filtered_df, "Box Plot des Salaires - Data Scientist", ax)
                st.pyplot(fig)

        elif granularité_selectionnee == "Région":
            liste_options = regions
            # Affichage du menu déroulant pour les régions
            regions_selectionnees = st.multiselect('Sélectionnez les régions', liste_options)

            # Filtrer le DataFrame pour inclure uniquement les régions sélectionnées
            filtered_df = df[df['region'].isin(regions_selectionnees)]


            # Pour les Boxplot
            # Filtrer les valeurs None
            double_filtered_df = filtered_df.dropna(subset=['salaire_annuel_mean'])
            #transformer salaire_annuel_mean en type int
            double_filtered_df['salaire_annuel_mean'] = double_filtered_df['salaire_annuel_mean'].astype(float)

            # Affichage du nombre total d'offres dans une boîte d'information
            st.info(f"**Nombre d'offres total : {len(filtered_df)}**")

            # Diviser la page en deux colonnes
            col1, col2 = st.columns(2)

            with col1:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres de Data Engineer: {len(filtered_df[filtered_df['poste']== 'data engineer'])}**")

                # Afficher le Wordcloud data Engineer dans la première colonne
                st.subheader("Wordcloud data Engineer :")
                corpus_data_engineer = preprocess_text_column(filtered_df[filtered_df['poste'] == 'data engineer']['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_engineer])
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png')
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_engineer, "Mots les plus fréquents - Data Engineer", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot Salaire data Engineer :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(double_filtered_df[double_filtered_df['poste'] == 'data engineer'], "Box Plot des Salaires - Data Engineer", ax)
                st.pyplot(fig)

            with col2:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres de data Scientist : {len(filtered_df[filtered_df['poste']== 'data scientist'])}**")

                # Afficher le Wordcloud data Scientist dans la deuxième colonne
                st.subheader("Wordcloud data Scientist :")
                corpus_data_scientist = preprocess_text_column(filtered_df[filtered_df['poste'] == 'data scientist']['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_scientist])
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png')
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_scientist, "Mots les plus fréquents - Data Scientist", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot Salaire data Scientist :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(double_filtered_df[double_filtered_df['poste'] == 'data scientist'], "Box Plot des Salaires - Data Scientist", ax)
                st.pyplot(fig)

        elif granularité_selectionnee == "Département":

            liste_options = departements
            # Affichage du menu déroulant
            departements_selectionnees = st.multiselect('Sélectionnez les départements', liste_options)

            # Filtrer le DataFrame pour inclure uniquement les régions sélectionnées
            filtered_df = df[df['departement_nom'].isin(departements_selectionnees)]

            # Pour les Boxplot
            # Filtrer les valeurs None
            double_filtered_df = filtered_df.dropna(subset=['salaire_annuel_mean'])
            #transformer salaire_annuel_mean en type int
            double_filtered_df['salaire_annuel_mean'] = double_filtered_df['salaire_annuel_mean'].astype(float)

            # Affichage du nombre total d'offres dans une boîte d'information
            st.info(f"**Nombre d'offres total : {len(filtered_df)}**")

            # Diviser la page en deux colonnes
            col1, col2 = st.columns(2)

            with col1:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres de data Engineer : {len(filtered_df[filtered_df['poste']== 'data engineer'])}**")

                # Afficher le Wordcloud data Engineer dans la première colonne
                st.subheader("Wordcloud data Engineer :")
                corpus_data_engineer = preprocess_text_column(filtered_df[filtered_df['poste'] == 'data engineer']['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_engineer])
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png')
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_engineer, "Mots les plus fréquents - Data Engineer", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot Salaire data Engineer :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(double_filtered_df[double_filtered_df['poste'] == 'data engineer'], "Box Plot des Salaires - Data Engineer", ax)
                st.pyplot(fig)

            with col2:

                # Affichage du nombre total d'offres dans une boîte d'information
                st.info(f"**Nombre d'offres de data scientist : {len(filtered_df[filtered_df['poste']== 'data scientist'])}**")

                # Afficher le Wordcloud data Scientist dans la deuxième colonne
                st.subheader("Wordcloud data Scientist :")
                corpus_data_scientist = preprocess_text_column(filtered_df[filtered_df['poste'] == 'data scientist']['profil'])
                texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_scientist])
                wordcloud = WordCloud(width=400, height=400, background_color='white', max_words=nombre_mots).generate(texte_concatene)
                img_buffer = BytesIO()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png')
                st.image(img_buffer.getvalue())

                # Afficher l'histogramme des mots les plus fréquents
                st.subheader("Histogramme:")
                fig, ax = plt.subplots()
                plot_most_common_words(corpus_data_scientist, "Mots les plus fréquents - Data Scientist", ax, max_words=nombre_mots)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # Afficher le box plot des salaires
                st.subheader("Boxplot Salaire data Scientist :")
                fig, ax = plt.subplots()
                plot_salary_boxplot(double_filtered_df[double_filtered_df['poste'] == 'data scientist'], "Box Plot des Salaires - Data Scientist", ax)
                st.pyplot(fig)