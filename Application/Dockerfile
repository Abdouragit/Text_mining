FROM python:3.11.7-slim

WORKDIR /app

# ajout des fichiers dans l'image
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
COPY Database.db /app/Database.db
COPY style /app/style
COPY logo /app/logo

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt && \
python -m nltk.downloader wordnet && \
python -m nltk.downloader stopwords && \
python -m nltk.downloader omw-1.4


# port
EXPOSE 8501

# lancement
CMD streamlit run app.py --server.port 8501 
