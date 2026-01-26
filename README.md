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


## Modules principaux

### `src/llm/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `src/database/database_creation.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

### `evaluation/generate_dataset.py`

Création des réponses par rapport aux questions déjà existantes.

### `evaluation/evaluate_ragas.py`

Evalue les performances de l'application en utilisant des questions et des réponses fournies.
[Évaluation d'un système RAG par Mistral](https://github.com/mistralai/cookbook/blob/main/mistral/evaluation/RAG_evaluation.ipynb)


## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `src/config/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

