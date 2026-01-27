# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Installer les dépendances**

```bash
uv sync
```

3. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
à rédiger

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `data/raw/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
uv run python3 src/llm/indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `data/raw/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `data/vector_store/`

### 3. Lancer l'application

```bash
uv run streamlit run src/app.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Structure du projet

### `src/llm/vector_store.py`

```
.
├── data                                # Données brutes et base de connaissances
├── evaluation
│   ├── evaluate_ragas.py               # Script d'évaluation
│   ├── generate_dataset.py             # Script de generation des réponses au fichier questions.csv
│   └── questions.csv
├── src
│   ├── nba_assistant
│   │   ├── app.py                      # Interface utilisateur Streamlit
│   │   ├── config
│   │   │   ├── config.py               # Paramètres personnalisables
│   │   ├── database
│   │   │   ├── database_creation.py    # Création de la base de données SQLite
│   │   │   ├── load_excel_to_db.py     # Chargement des données depuis un fichier Excel
│   │   │   ├── nba_db.db               # Base de données SQLite
│   │   │   └── schemas.py              # Schemas Pydantic pour la base de données
│   │   ├── llm
│   │   │   ├── data_loader.py          # Chargement des données à partir de divers fichiers
│   │   │   ├── indexer.py              # Script d'indexation des données avec FAISS
│   │   │   ├── llm.py                  # Classe RAGAgent
│   │   │   ├── sql_tool.py             # Tool pour la gestion de la base de données
│   │   │   ├── vector_store_management.py  # Gestion de l'index FAISS
│   │   │   └── vector_store_tool.py     # Tool pour la recherche sémantique
│   │   └── utils
│   │       ├── logging_handler.py      # Configuration du logging
```
Evalue les performances de l'application en utilisant des questions et des réponses fournies.
[Évaluation d'un système RAG par Mistral](https://github.com/mistralai/cookbook/blob/main/mistral/evaluation/RAG_evaluation.ipynb)


## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `src/config/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

