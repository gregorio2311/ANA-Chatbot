"""
CONFIGURACIÓN DEL MODELO BGE - ANA-CHATBOT
==========================================

Este script define la configuración del modelo de embeddings utilizado en todo el proyecto.
El modelo BAAI/bge-large-en-v1.5 es un modelo de última generación para embeddings semánticos.

CARACTERÍSTICAS DEL MODELO:
- Nombre: BAAI/bge-large-en-v1.5
- Dimensión: 1024 vectores
- Idioma: Inglés (funciona bien con español médico)
- Arquitectura: Transformer-based
- Rendimiento: Optimizado para búsqueda semántica

USO:
    from scripts.modelo_BGE import model
    
    # Generar embeddings
    embeddings = model.encode(["texto de ejemplo"])
    
    # Búsqueda semántica
    query_embedding = model.encode("consulta de búsqueda")

"""

from sentence_transformers import SentenceTransformer

# Modelo de embeddings semánticos para búsqueda
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
