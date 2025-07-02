"""
CHATBOT DE ANATOMÍA - SISTEMA DE BÚSQUEDA SEMÁNTICA
====================================================

Este script implementa un chatbot especializado en anatomía que utiliza búsqueda semántica
para responder consultas sobre contenido médico-anatómico almacenado en Pinecone.

FUNCIONALIDADES:
- Búsqueda semántica en documentos de anatomía
- Filtrado por libro específico
- Interfaz interactiva de consultas
- Visualización de resultados con puntuaciones

REQUISITOS:
- Variables de entorno configuradas (.env):
  * PINECONE_API_KEY: Clave API de Pinecone
  * PINECONE_HOST: URL del host de Pinecone
  * PINECONE_INDEX_NAME: Nombre del índice (opcional, default: "ana")

USO:
    python scripts/consulta.py

EJEMPLO DE USO:
    $ python scripts/consulta.py
    📚 Libros disponibles: G_A_S_4_E, LIBRO_IFSSA
    🔍 CHATBOT DE ANATOMÍA
    ============================================================
    Ingrese su consulta: ¿Qué es el sistema nervioso central?
    
    --- Resultado 1 ---
    Puntuación: 0.892
    Libro: G_A_S_4_E
    Fragmento ID: G_A_S_4_E_fragmento_0045
    Palabras: 487
    Texto: El sistema nervioso central está compuesto por...

DEPENDENCIAS:
- sentence-transformers: Para generar embeddings semánticos
- pinecone: Para búsqueda vectorial
- python-dotenv: Para cargar variables de entorno

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

import os
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class SemanticSearch:
    """
    Clase para realizar búsquedas semánticas en documentos de anatomía.
    
    Esta clase maneja la conexión con Pinecone, la generación de embeddings
    y la búsqueda semántica de contenido médico.
    
    Attributes:
        api_key (str): Clave API de Pinecone
        host (str): URL del host de Pinecone
        index_name (str): Nombre del índice en Pinecone
        model (SentenceTransformer): Modelo para generar embeddings
        index (Pinecone.Index): Índice de Pinecone para búsquedas
    """
    
    def __init__(self):
        """
        Inicializa el sistema de búsqueda semántica.
        
        Configura la conexión con Pinecone y carga el modelo de embeddings.
        Valida que todas las variables de entorno necesarias estén configuradas.
        
        Raises:
            ValueError: Si faltan variables de entorno requeridas
            Exception: Si hay errores en la inicialización
        """
        try:
            # Configuración desde variables de entorno
            self.api_key = os.getenv("PINECONE_API_KEY")
            self.host = os.getenv("PINECONE_HOST")
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "ana")
            
            if not all([self.api_key, self.host, self.index_name]):
                raise ValueError("Faltan variables de entorno: PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME")
            
            # Inicializar modelo
            logger.info("Cargando modelo de embeddings...")
            self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            
            # Inicializar Pinecone
            logger.info("Conectando a Pinecone...")
            pc = Pinecone(api_key=self.api_key)
            
            # Validar existencia del índice
            indexes = pc.list_indexes()
            index_names = [idx["name"] for idx in indexes]
            logger.info(f"Índices disponibles: {index_names}")
            
            if self.index_name not in index_names:
                raise ValueError(f"El índice '{self.index_name}' no existe en Pinecone. Índices disponibles: {index_names}")
            
            self.index = pc.Index(self.index_name)
            logger.info(f"Conectado al índice '{self.index_name}' correctamente")
            
            # Obtener estadísticas del índice
            stats = self.index.describe_index_stats()
            logger.info(f"Estadísticas del índice: {stats}")
            
        except Exception as e:
            logger.error(f"Error al inicializar el sistema: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, libro_filtro: str | None = None) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda semántica en el índice de Pinecone.
        
        Args:
            query (str): Texto de la consulta a buscar
            top_k (int, opcional): Número máximo de resultados a retornar. Default: 5
            libro_filtro (str, opcional): Filtrar resultados por libro específico. Default: None
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados con scores y metadatos
            
        Raises:
            Exception: Si hay errores en la búsqueda
            
        Example:
            >>> searcher = SemanticSearch()
            >>> results = searcher.search("sistema nervioso", top_k=3, libro_filtro="G_A_S_4_E")
            >>> print(f"Encontrados {len(results)} resultados")
        """
        try:
            logger.info(f"Buscando: '{query}'")
            if libro_filtro:
                logger.info(f"Filtro de libro: '{libro_filtro}'")
            
            # Codificar la consulta
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Realizar búsqueda
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k * 2 if libro_filtro else top_k,  # Buscar más si hay filtro
                include_metadata=True
            )
            
            # Filtrar por libro si se especifica
            if libro_filtro:
                filtered_results = []
                for match in results["matches"]:
                    if match["metadata"].get("libro") == libro_filtro:
                        filtered_results.append(match)
                        if len(filtered_results) >= top_k:
                            break
                results["matches"] = filtered_results
            
            logger.info(f"Encontrados {len(results['matches'])} resultados")
            return results["matches"]
            
        except Exception as e:
            logger.error(f"Error en la búsqueda: {e}")
            raise
    
    def get_available_books(self) -> List[str]:
        """
        Obtiene la lista de libros disponibles en el índice.
        
        Returns:
            List[str]: Lista de nombres de libros únicos
            
        Example:
            >>> searcher = SemanticSearch()
            >>> libros = searcher.get_available_books()
            >>> print(f"Libros disponibles: {libros}")
        """
        try:
            # Realizar una búsqueda vacía para obtener metadatos
            results = self.index.query(
                vector=[0.0] * 1024,  # Vector vacío
                top_k=1,
                include_metadata=True
            )
            
            # Obtener todos los libros únicos
            libros = set()
            for match in results["matches"]:
                libro = match["metadata"].get("libro")
                if libro:
                    libros.add(libro)
            
            return list(libros)
            
        except Exception as e:
            logger.error(f"Error obteniendo libros disponibles: {e}")
            return []
    
    def display_results(self, matches: List[Dict[str, Any]]):
        """
        Muestra los resultados de búsqueda de forma legible en consola.
        
        Args:
            matches (List[Dict[str, Any]]): Lista de resultados de búsqueda
            
        Example:
            >>> results = searcher.search("anatomía")
            >>> searcher.display_results(results)
        """
        if not matches:
            print("No se encontraron resultados.")
            return
            
        for i, match in enumerate(matches, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Puntuación: {match['score']:.3f}")
            print(f"Libro: {match['metadata'].get('libro', 'Desconocido')}")
            print(f"Fragmento ID: {match['metadata'].get('fragmento_id', 'N/A')}")
            print(f"Palabras: {match['metadata'].get('palabras', 0)}")
            print(f"Texto: {match['metadata']['texto']}")
            print("-" * 60)

def main():
    """
    Función principal que ejecuta el chatbot interactivo.
    
    Esta función:
    1. Inicializa el sistema de búsqueda
    2. Muestra los libros disponibles
    3. Ejecuta un bucle interactivo para consultas
    4. Permite filtrar por libro específico
    5. Muestra resultados formateados
    
    Para salir del programa, escribe 'salir' cuando se solicite la consulta.
    """
    try:
        # Crear instancia del buscador
        searcher = SemanticSearch()
        
        # Mostrar libros disponibles
        libros_disponibles = searcher.get_available_books()
        if libros_disponibles:
            print(f"📚 Libros disponibles: {', '.join(libros_disponibles)}")
        else:
            print("⚠️ No se pudieron obtener los libros disponibles")
        
        while True:
            print("\n" + "="*60)
            print("🔍 CHATBOT DE ANATOMÍA")
            print("="*60)
            
            # Consulta de ejemplo (puede ser interactiva)
            query = input("Ingrese su consulta: ")
            if query.lower() == "salir":
                break
            
            # Opción de filtrar por libro
            print("\nOpciones de búsqueda:")
            print("1. Buscar en todos los libros")
            for i, libro in enumerate(libros_disponibles, 2):
                print(f"{i}. Buscar solo en '{libro}'")
            
            try:
                opcion = input(f"\nSeleccione opción (1-{len(libros_disponibles)+1}): ")
                opcion = int(opcion)
                
                if opcion == 1:
                    libro_filtro = None
                elif 2 <= opcion <= len(libros_disponibles) + 1:
                    libro_filtro = libros_disponibles[opcion - 2]
                else:
                    print("Opción inválida, buscando en todos los libros...")
                    libro_filtro = None
                    
            except ValueError:
                print("Opción inválida, buscando en todos los libros...")
                libro_filtro = None
            
            # Realizar búsqueda
            results = searcher.search(query, top_k=3, libro_filtro=libro_filtro)
        
            # Mostrar resultados
            searcher.display_results(results)
        
        
        
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
