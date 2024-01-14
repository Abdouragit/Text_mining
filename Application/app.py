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
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
from wordcloud import WordCloud
import plotly.express as px
from io import BytesIO
ponctuations = list(string.punctuation)
nltk.download('omw-1.4')
from collections import Counter
import seaborn as sns

# histogramme
def plot_most_common_words(corpus, title, ax, max_words=10):
    flat_list = [word for sublist in corpus for word in sublist]
    word_freq = Counter(flat_list)
    most_common = dict(word_freq.most_common(max_words))
    ax.bar(most_common.keys(), most_common.values())
    ax.set_title(title)
    ax.set_xlabel('Mots')
    ax.set_ylabel('Fréquence')

# fonction pour créer un box plot des salaires
def plot_salary_boxplot(data, title, ax):
    sns.boxplot(x='poste', y='salaire_annuel_mean', data=data, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Poste')
    ax.set_ylabel('Salaire Annuel (en euros)')

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
    mots_frequents = Counter([mot for phrase in corpus for mot in phrase])
    mots_frequents = mots_frequents.most_common()
    top_terms = [mot[0] for mot in mots_frequents[:nombre_mots_a_afficher]] if nombre_mots_a_afficher is not None else [mot[0] for mot in mots_frequents]
    modele = Word2Vec(corpus, vector_size=vector_size, window=window)
    words = modele.wv
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
        offres.id_entreprise,
        d_ville.ville,
        d_ville.latitude,
        d_ville.longitude,
        d_ville.code_postal,
        d_ville.departement,
        d_entreprise.entreprise_nom,
        d_entreprise.entreprise_description,
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
onglets = ["Accueil", "Statistiques générales", "Cartographies", "Analyse du corpus"]

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
        c1, c2, c3 = st.columns((0.45, 0.038, 0.045))
        #Colonne de droite pour l'image
        with c2:
            st.write("##")
            st.write("##")
            st.write("##")
            st.image("logo/python.png", width=70)
            st.write("##")
            st.image("logo/streamlit.png", width=70)
            st.write("##")
        
        with c3:
            st.write("##")
            st.write("##")
            st.write("##")
            st.image("logo/sqlite.png", width=150)
            st.image("logo/plotly.png", width=150)
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
            # Use local CSS
            def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


            local_css("style/style.css")
            # Load Animation
            animation_symbol = "❄"

            st.markdown(
                f"""
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                <div class="snowflake">{animation_symbol}</div>
                """,
                unsafe_allow_html=True,
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

    elif onglet_selectionne == "Cartographies":
        # Function to create Folium map with markers and dynamic count
        def create_folium_map_dynamic_count(df, map_title = ''):
            map_center = [df['latitude'].mean(), df['longitude'].mean()]
            my_map = folium.Map(location=map_center, zoom_start=5)

            marker_cluster = MarkerCluster().add_to(my_map)

            for _, row in df.iterrows():
                folium.Marker([row['latitude'], row['longitude']], popup=row['intitule'] + '  ' + row['ville'] + '  ' + ' ' + str(row['entreprise_nom']) if row['entreprise_nom'] is not None else ''  ).add_to(marker_cluster)

            st.header(map_title)

            # Use folium_static to display Folium map in Streamlit
            folium_static(my_map)
        

        def filter_dataframe(df, filter_column, filter_values):
            """Filter the DataFrame based on selected values in a specific column."""
            return df[df[filter_column].isin(filter_values)]

        # Function to create Folium heatmap with filtering and dynamic count
        def create_folium_heatmap_dynamic_count(df, selected_filter_column, selected_filter_values, map_title = ""):
            filtered_df = filter_dataframe(df, selected_filter_column, selected_filter_values)

            if not filtered_df.empty:
                map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
                my_map = folium.Map(location=map_center, zoom_start=5)

                heat_data = [[row['latitude'], row['longitude']] for _, row in filtered_df.iterrows()]
                HeatMap(heat_data, max_val=len(filtered_df['intitule'])).add_to(my_map)
                
                my_map.add_child(folium.LinearColormap(['green', 'yellow', 'red'], vmin=2, vmax=df['intitule'].value_counts().max()).add_to(my_map))
                st.header(map_title)

                # Use folium_static to display Folium map in Streamlit
                folium_static(my_map)
            else:
                st.warning(f"No data available for the selected {selected_filter_column}")

        # Map 1: Distribution et concentration des offres
        st.header("Map 1: Distribution des offres d'emploi")

        # Creation des filtres pour Map 1
        selected_filter_1 = st.selectbox("Selectionner le niveau de filtrage (Map 1)", ["Region", "Departement", "Ville"], key="selected_filter_1", index=0)
        if selected_filter_1 == "Region":
            filter_options_1 = df['region'].unique()
        elif selected_filter_1 == "Departement":
            filter_options_1 = df['departement'].unique()
        elif selected_filter_1 == "Ville":
            filter_options_1 = df['ville'].unique()

        selected_filter_values_1 = st.multiselect(f"Selection des {selected_filter_1}s ", filter_options_1, default=filter_options_1)

        # Apply filters to create filtered DataFrame for Map 1
        filtered_df_1 = df[df[selected_filter_1.lower()].isin(selected_filter_values_1)]

        # Create Folium map with markers for Map 1
        create_folium_map_dynamic_count(filtered_df_1,"Distribution des offres d'emploi")


        # Map 2: Visualisation de l'ensemble des offres d'emplois
        st.header("Map 2: Visualisation de l'ensemble des offres d'emplois")

        map_center_2 = [df['latitude'].mean(), df['longitude'].mean()]
        my_map_2 = folium.Map(location=map_center_2, zoom_start=5)

        selected_filter_2 = st.selectbox("Selectionner le niveau de filtrage", ["Region", "Departement", "Ville"], key="selected_filter_2", index=0)
        # print('selected_filter_2')
        # print(selected_filter_2)
        if selected_filter_2 == "Region":
            filter_options_2 = df['region'].unique()
        elif selected_filter_2 == "Departement":
            filter_options_2 = df['departement'].unique()
        elif selected_filter_2 == "Ville":
            filter_options_2 = df['ville'].unique()

        selected_filter_values_2 = st.multiselect(f"Selection des {selected_filter_2}s", filter_options_2, default=filter_options_2)

        filtered_df_2 = df[df[selected_filter_2.lower()].isin(selected_filter_values_2)]

        for _, row in filtered_df_2.iterrows():
            folium.Marker([row['latitude'], row['longitude']], popup=row['intitule'] + '  ' + row['ville'] + '  ' + ' ' + str(row['entreprise_nom']) if row['entreprise_nom'] is not None else ''  ).add_to(my_map_2)
        
        st.header("Visualisation de l'ensemble des offres d'emplois")

        # display Folium map in Streamlit
        folium_static(my_map_2)

        # Map 3 Heatmap - Concentration des offres d'emploi
        st.header("Map 3: Heatmap - Concentration des offres d'emploi")

        # Definition de filter_options 
        select_all_button_3 = st.button("Select All", key="select_all_button_3")
        deselect_all_button_3 = st.button("Deselect All", key="deselect_all_button_3")
        selected_filter_3 = st.selectbox("Selectionner le niveau de filtrage", ["Region", "Departement", "Ville"], key='selected_filter_3')

        filter_options_3 = df[selected_filter_3.lower()].unique()

        # print('filter_options_3 :')
        # print(filter_options_3)
        # print('selected_filter_3 :')
        # print(selected_filter_3)
        selected_filter_values_3 = filter_options_3

        if select_all_button_3:
            selected_filter_values_3 = df[selected_filter_3.lower()].unique()
        elif deselect_all_button_3:
            selected_filter_values_3 = []
        else:
            selected_filter_values_3 = st.multiselect(f"Selection les {selected_filter_3}s ", filter_options_3, default=filter_options_3,)

        create_folium_heatmap_dynamic_count(df, selected_filter_3.lower(), selected_filter_values_3, "Heatmap - Concentration des offres d'emploi")

    # Afficher le tableau de bord sélectionné
    elif onglet_selectionne == "Statistiques générales":

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
            regions_selectionnees = st.multiselect('Sélectionnez les régions', liste_options, default = liste_options)

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
            departements_selectionnees = st.multiselect('Sélectionnez les départements', liste_options, default = liste_options)

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