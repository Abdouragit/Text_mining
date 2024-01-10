import streamlit as st
import sqlite3
import pandas as pd

# Fonction pour charger les données depuis la base de données
def charger_donnees():
    # accès à la table departement pour afficher son contenu
    connexion = sqlite3.connect('Database.db')
    query = '''SELECT 
        offres.intitule,
        offres.description_offre,
        offres.date_creation,
        offres.salaire_min_annuel,
        offres.salaire_max_annuel,
        offres.salaire_annuel_mean,
        offres.qualification_libelle,
        offres.experience,
        offres.type_contrat,
        offres.secteur_activite,
        d_ville.ville,
        d_ville.latitude,
        d_ville.longitude,
        d_ville.code_postal,
        d_ville.departement AS ville_departement,
        d_entreprise.entreprise_nom,
        d_entreprise.entreprise_description,
        h_departement.departement,
        h_departement.departement_nom,
        h_departement.region
            
        FROM offres
        JOIN d_ville ON offres.id_ville = d_ville.id
        JOIN d_entreprise ON offres.id_entreprise = d_entreprise.id
        JOIN h_departement ON d_ville.departement = h_departement.departement;
    '''
    df = pd.read_sql_query(query, connexion)
    connexion.close()
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
