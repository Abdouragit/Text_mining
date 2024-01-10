import streamlit as st
import sqlite3
import pandas as pd

# Fonction pour charger les données depuis la base de données
def charger_donnees():
    conn = sqlite3.connect('Database.db')
    query = 'SELECT * FROM d_ville'
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Chargement des données
df = charger_donnees()

# Première "fenêtre"
with st.sidebar:
    st.title("Sommaire - On va mettre ici toutes les 'pages' de dashboard cliquables")

    # Boutons pour différents tableaux de bord
    bouton_dashboard1 = st.button("Dashboard 1")
    bouton_dashboard2 = st.button("Dashboard 2")

# Deuxième "fenêtre"
with st.container():
    st.title("Fenêtre 2 - Contenu principal")

    # Afficher le Dashboard 1 si le bouton est cliqué
    if bouton_dashboard1:
        st.subheader("Dashboard 1 :")
        
        # Afficher le DataFrame dans le Dashboard 1
        st.subheader("Données depuis la base de données :")
        st.dataframe(df)

        # Vous pouvez ajouter d'autres composants ou actions spécifiques au Dashboard 1 ici
