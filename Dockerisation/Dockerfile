FROM python:3.11.7-slim

WORKDIR /app

# ajout des fichiers dans l'image
ADD app.py /app/app.py
ADD requirements.txt /app/requirements.txt
ADD Database.db /app/Database.db

RUN pip install -r requirements.txt

# port
EXPOSE 8502

# lancement
CMD ["streamlit", "run", "--server.port", "8502" ,"app.py"]
