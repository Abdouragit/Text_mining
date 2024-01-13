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
    # st.image("logo/job.png")
    st.title("Sommaire")

    # Utiliser le cache pour les onglets
    onglet_selectionne = st.sidebar.radio("Sélectionnez un onglet ", onglets, format_func=lambda x:x)

# Deuxième "fenêtre"
with st.container():
    st.title(f"{onglet_selectionne}")

    # Afficher le tableau de bord sélectionné
    if onglet_selectionne == "Cartographie":
        import folium
        from streamlit_folium import folium_static
        from folium.plugins import HeatMap, MarkerCluster
        from wordcloud import WordCloud
        import plotly.express as px

        # Access the 'departement' table to display its content
        connexion = sqlite3.connect('Database.db')
        cursor = connexion.cursor()

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

        df = pd.read_sql_query(query, connexion)

        connexion.close()


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


        # # Function to create Word Cloud for a specific department
        # def create_word_cloud(department_df, department_name):
        #     st.subheader(f"Word Cloud for {department_name}")

        #     wordcloud = WordCloud(width=800, height=400).generate(' '.join(department_df['description_offre']))

        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(wordcloud, interpolation='bilinear')
        #     plt.axis('off')
        #     st.pyplot(plt)

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
            selected_filter_values_3 = st.multiselect(f"Selection des {selected_filter_3}s ", filter_options_3, default=filter_options_3)

        create_folium_heatmap_dynamic_count(df, selected_filter_3.lower(), selected_filter_values_3, "Heatmap - Concentration des offres d'emploi")

    elif onglet_selectionne == "Dashboard 2":
        st.subheader("Dashboard 2 :")
        # Ajoutez ici le contenu spécifique au Dashboard 2

