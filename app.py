import streamlit as st
import sqlite3
import pandas as pd

# Fonction pour charger les données depuis la base de données
@st.cache_resource
def charger_donnees():
    conn = sqlite3.connect('Database.db')
    query = 'SELECT * FROM d_ville'
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Chargement des données
df = charger_donnees()


# Liste des onglets
onglets = ["Accueil", "Statistiques", "Cartographie", "Analyse du corpus"]

# Première "fenêtre"
with st.sidebar:
    st.title("Sommaire - On va mettre ici toutes les 'pages' de dashboard cliquables")

    # Utiliser le cache pour les onglets
    onglet_selectionne = st.selectbox("Sélectionnez un dashboard :", onglets, format_func=lambda x: x)

# Deuxième "fenêtre"
with st.container():
    st.title(f"Fenêtre 2 - Contenu principal - {onglet_selectionne}")

    # Utiliser le cache pour éviter le rechargement des données à chaque interaction
    df = charger_donnees_cache()

    # Afficher le tableau de bord sélectionné
    if onglet_selectionne == "Dashboard 1":
        st.subheader("Dashboard 1 :")
        # Ajoutez ici le contenu spécifique au Dashboard 1
    elif onglet_selectionne == "Dashboard 2":
        st.subheader("Dashboard 2 :")
        # Ajoutez ici le contenu spécifique au Dashboard 2

    # Afficher le DataFrame dans chaque tableau de bord
    st.subheader("Données depuis la base de données :")
    st.dataframe(df)
