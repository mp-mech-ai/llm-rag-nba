from pydantic import BaseModel, Field
from langchain_community.tools import tool
from nba_assistant.llm.vector_store_management import create_vector_store_manager
import logging
from nba_assistant.config.config import SYSTEM_PROMPT
from langchain_community.tools import BaseTool
from typing import Type

class VectorStoreInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="The query to search for")
    nb_results: int = Field(..., gt=0, le=10, description="Number of results to return")

# class VectorStoreResearchTool(BaseTool):
#     name : str = "vector_store_research"
#     description : str = (
#         "Utilise cet outil pour faire une recherche dans la base de connaissance portant sur la saison NBA."
#         "La 'query' doit être concise mais exacte afin de pouvoir récupérer les informations les plus pertinentes."
#         "Le 'nb_results' doit etre un nombre entre 1 et 10, et il definit le nombre de documents renvoyés par la recherche."
#         "Par exemple, si l'utilisateur demande 'Quelle équipe à la meilleur défense', alors la 'query' sera 'Équipe avec la meilleure defense'"
#     )
#     args_schema: Type[BaseModel] = VectorStoreInput

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs) 
#         self.vector_store = create_vector_store_manager()

#     def _run(self, query: str, nb_results: int) -> list:
#         try:
#             search_results = self.vector_store.search(query, k=nb_results)
#         except Exception as e:
#             logging.exception("Recherche dans la base de connaissance echouée")
#             return [f"Recherche dans la base de connaissance echouée: {e}"]
        
#         return search_results

#     async def _arun(self, query: str):
#         raise NotImplementedError

@tool
def vector_store_research(query_param:  VectorStoreInput) -> str:
    """
    Utilise cet outil pour faire une recherche dans la base de connaissance portant sur la saison NBA.
    La "query" doit être concise (moins de 100 caractères) mais exacte afin de pouvoir récupérer les informations les plus pertinentes.
    Le "nb_results" doit etre un nombre entre 1 et 10, et il definit le nombre de documents renvoyés par la recherche.

    Par exemple, si l'utilisateur demande "Quelle équipe à la meilleur défense", alors la "query" sera "Équipe avec la meilleure defense"
    """
    query = query_param.query
    nb_results = query_param.nb_results

    vector_store = create_vector_store_manager()
    try:
        search_results = vector_store.search(query, k=nb_results)
    except Exception as e:
        logging.exception("Recherche dans la base de connaissance echouée")
        return f"Recherche dans la base de connaissance echouée: {e}"

    context_str = "\n\n---\n\n".join(
        f"Source: {res['metadata'].get('source', 'Inconnue')}\n"
        f"Contenu: {res['text']}"
        for res in search_results
    )

    return context_str
