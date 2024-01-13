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
#liste des ponctuations
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

# Chargement des données
df = charger_donnees()

#dataframe pour data-engineer
data_engineer_df = df[df['poste']== 'data engineer']

#dataframe pour data-scientist
data_scientist_df = df[df['poste']== 'data scientist']


# Première "fenêtre"
with st.sidebar:
    st.title("Sommaire - On va mettre ici toutes les 'pages' de dashboard cliquables")

    # Boutons pour différents tableaux de bord
    bouton_dashboard1 = st.button("Dashboard 1")
    bouton_dashboard2 = st.button("Dashboard 2")

# Deuxième "fenêtre"
with st.container():

    # Afficher le Dashboard 1 si le bouton est cliqué
    # Afficher le Dashboard 1 si le bouton est cliqué
    if bouton_dashboard1:
        st.subheader("Dashboard 1 :")
        
        # Diviser la page en deux colonnes
        col1, col2 = st.columns(2)
        
        # Afficher le Wordcloud data Engineer dans la première colonne
        # Afficher le Wordcloud data Engineer dans la première colonne
        with col1:
            st.subheader("Wordcloud data Engineer :")
            #corpus nettoyé de data ingé
            corpus_data_engineer = preprocess_text_column(data_engineer_df['profil'])
            texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_engineer])
            # Créer l'objet WordCloud
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate(texte_concatene)
            
            # Utiliser BytesIO pour enregistrer l'image
            img_buffer = BytesIO()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Masquer les axes
            plt.savefig(img_buffer, format='png')
            
            # Afficher l'image avec st.image()
            st.image(img_buffer.getvalue())

        # Afficher le Wordcloud data Scientist dans la deuxième colonne
        with col2:
            st.subheader("Wordcloud data Scientist :")
            #corpus nettoyé de data scientist
            corpus_data_scientist = preprocess_text_column(data_scientist_df['profil'])
            texte_concatene = ' '.join([' '.join(doc) for doc in corpus_data_scientist])
            # Créer l'objet WordCloud
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate(texte_concatene)
            
            # Utiliser BytesIO pour enregistrer l'image
            img_buffer = BytesIO()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Masquer les axes
            plt.savefig(img_buffer, format='png')
            
            # Afficher l'image avec st.image()
            st.image(img_buffer.getvalue())
