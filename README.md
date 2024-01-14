# Text_mining

## Importation de l'image Docker et lancement de l'application

### Étape 1: Importation de l'image Docker

Assurez-vous que Docker-desktop est installé sur votre machine. Si ce n'est pas le cas, téléchargez et installez Docker depuis [le site officiel de Docker](https://www.docker.com/get-started).

Ouvrez un terminal et exécutez la commande suivante pour télécharger l'image Docker de l'application :

'''bash
docker pull abdouragit/nlpapp:1.0
'''

### Étape 2: Lancement de l'application

Exécutez l'application Streamlit dans un conteneur Docker en utilisant la commande suivante :

'''bash
docker run --rm -p 8501:8501 -it abdouragit/nlpapp:1.0
'''

Cette commande démarre le conteneur Docker et redirige le port 8501 de votre machine vers le port 8501 du conteneur, où l'application Streamlit est en cours d'exécution.

### Étape 3: Accéder à l'application

Ouvrez votre navigateur web et saisissez l'URL suivante dans la barre de navigation :

[http://localhost:8501](http://localhost:8501)

Vous devriez maintenant pouvoir explorer l'application Streamlit depuis votre navigateur.

Note: Assurez-vous que le port 8501 est disponible et n'est pas utilisé par une autre application sur votre machine. Si le port est déjà utilisé, vous pouvez utiliser un autre port lors de la commande `docker run`, par exemple `-p 8080:8501`.