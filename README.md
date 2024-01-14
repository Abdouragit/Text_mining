# Text_mining
## Présentation

Bienvenue sur la page de notre projet de Text Mining, réalisé par Natacha Perez, Martin Revel et Abdourahmane Ndiaye. Nous mettons ici à disposition le code permettant d'utiliser notre application pour faire des analyses d'offres d'emplois scrappés sur les sites de Pôle emploi et de l'Apec.

Pour plus d'informations sur le contenu du projet en général, notemment la mise en conteneur docker vous pourrez vous referer au rapport suivant: [http://localhost:8501](http://localhost:8501)

Vous pourrez trouver dans [Scripts_ETL](http://localhost:8501) les codes ayant permis de scrapper les offres d'emploi, leur nettoyage ainsi que la création de la base de donnée.

Vous pourrez également trouver dans [Application](http://localhost:8501) le code de l'application streamlit ainsi que la base de donnée pysqlite3.

## Importation de l'image Docker et lancement de l'application

### Étape 1: Importation de l'image Docker

Assurez-vous que Docker-desktop est installé sur votre machine. Si ce n'est pas le cas, téléchargez et installez Docker depuis [le site officiel de Docker](https://www.docker.com/get-started).

Ouvrez un terminal et exécutez la commande suivante pour télécharger l'image Docker de l'application :

```
docker pull abdouragit/nlpapp:1.0
```

### Étape 2: Lancement de l'application

Exécutez l'application Streamlit dans un conteneur Docker en utilisant la commande suivante :

```bash
docker run --rm -p 8501:8501 -it abdouragit/nlpapp:1.0
```

Cette commande démarre le conteneur Docker et redirige le port 8501 de votre machine vers le port 8501 du conteneur, où l'application Streamlit est en cours d'exécution.

### Étape 3: Accéder à l'application

Ouvrez votre navigateur web et saisissez l'URL suivante dans la barre de navigation :

[http://localhost:8501](http://localhost:8501)

Vous devriez maintenant pouvoir explorer l'application Streamlit depuis votre navigateur.

Note: Assurez-vous que le port 8501 est disponible et n'est pas utilisé par une autre application sur votre machine. Si le port est déjà utilisé, vous pouvez utiliser un autre port lors de la commande `docker run`, par exemple `-p 8080:8501`.

liens youtube des tutoriels videos: 
https://youtu.be/F_X4Phu80D8
https://youtu.be/l38fpcoPPv0