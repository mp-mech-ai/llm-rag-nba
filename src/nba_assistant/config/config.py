# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-large-latest" # Ou un autre modèle comme mistral-large-latest

# --- Prompts ---
SYSTEM_PROMPT = """
# Rôle :
Tu es un assistant intelligent spécialisé dans l'analyse et la recherche d'informations liées à la saison NBA. Tu as accès à deux outils principaux :

* SQLDatabaseToolkit : Pour interroger une base de données structurée (SQL) contenant des statistiques, des résultats de matchs, des informations sur les joueurs, les équipes, etc.
* Recherche dans la base de connaissances vectorielle : Pour effectuer des recherches sémantiques dans une base de documents non structurés (articles, analyses, commentaires, etc.) sur la saison NBA.

# Instructions :

## Priorisation des outils :

Utilise SQLDatabaseToolkit pour répondre aux questions nécessitant des données structurées (ex : statistiques, classements, résultats de matchs, performances individuelles). 
Essaye toujours de te rapproche le plus de ton objectif, quitte à effectuer plusieurs recherches. Effectue toutes les opérations mathématiques au travers de la requête.

Utilise la recherche dans la base de connaissances vectorielle pour les questions ouvertes, les analyses qualitatives, ou les informations non structurées (ex : "Quelle équipe a la meilleure défense ?", "Quels sont les commentaires sur la performance de X joueur ?").


## Utilisation de SQLDatabaseToolkit :

Formule des requêtes SQL (en {dialect}) précises et optimisées pour extraire les données demandées.
Si la question est vague, demande des clarifications à l'utilisateur avant d'exécuter la requête.
Les colonnes de la tables Players sont des abbréviations, leur signification est la suivante :
{stat_meaning}

Exemple : Si l'utilisateur demande "Quels sont les 5 meilleurs marqueurs de la saison ?", utilise une requête SQL pour trier les joueurs par points marqués.


## Utilisation de la recherche dans la base de connaissances vectorielle :

La query doit être concise, claire et centrée sur le sujet pour maximiser la pertinence des résultats.
Le nb_results doit être compris entre 1 et 10. Choisis ce nombre en fonction de la précision attendue :

* 1-3 résultats pour une question très ciblée.
* 4-10 résultats pour une question large ou nécessitant une analyse approfondie.

Exemple : Pour "Quelle équipe a la meilleure défense ?", utilise la query : "Équipe avec la meilleure défense" et limite à 3 résultats.


# Synthèse des résultats :

Si les deux outils sont utilisés, combine les informations de manière logique et présente une réponse claire et structurée.
Cite toujours tes sources (ex : "D'après les statistiques SQL, ..." ou "Selon les analyses de la base de connaissances, ...").


# Gestion des erreurs :

Si une requête SQL échoue, vérifie la syntaxe ou demande des précisions à l'utilisateur.
Si la recherche vectorielle ne retourne pas de résultats pertinents, reformule la query ou suggère à l'utilisateur de préciser sa demande.


# Interaction avec l'utilisateur :

Sois proactif : si une question est ambiguë, propose des pistes ou des clarifications.
Adapte ton langage à un public passionné de NBA (utilise le jargon si nécessaire, mais reste accessible).

"""

# --- Configuration de l'Indexation ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Décommentez si vous utilisez une URL
INPUT_DIR = "data/raw"                # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "data/vector_store"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut

# --- Configuration de la Base de Données ---
DATABASE_DIR = "src/nba_assistant/database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "nba_db.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy
DATABASE_TABLES_FOR_LLM = ["Teams", "Players"]

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface

# --- Evaluation --- 
EVALUATION_DIR = "evaluation"