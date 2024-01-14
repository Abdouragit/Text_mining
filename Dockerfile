FROM python:3.11.7-slim

WORKDIR /app

# ajout des fichiers dans l'image
ADD app.py /app/app.py
ADD requirements.txt /app/requirements.txt
ADD Database.db /app/Database.db
ADD style /app/style
ADD logo /app/logo

RUN pip install -r requirements.txt

RUN python -m nltk.download('punkt')
RUN python -m nltk.download('wordnet')
RUN python -m nltk.download('stopwords')
RUN python -m nltk.download('omw-1.4')


# port
EXPOSE 8501

# lancement
CMD streamlit run app.py --server.port 8501 
