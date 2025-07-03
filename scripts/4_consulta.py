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
import json
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
        fragmentos_originales (List[Dict]): Fragmentos originales cargados del JSON
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
            
            # Cargar fragmentos originales desde JSON
            self.cargar_fragmentos_originales()
            
        except Exception as e:
            logger.error(f"Error al inicializar el sistema: {e}")
            raise
    
    def cargar_fragmentos_originales(self):
        """
        Carga los fragmentos originales desde el archivo JSON para recuperar el texto completo.
        """
        try:
            # Buscar el archivo JSON en diferentes ubicaciones posibles
            posibles_rutas = [
                "data/fragmentos_metadata.json",
                "../data/fragmentos_metadata.json",
                "scripts/../data/fragmentos_metadata.json"
            ]
            
            fragmentos_file = None
            for ruta in posibles_rutas:
                if os.path.exists(ruta):
                    fragmentos_file = ruta
                    break
            
            if not fragmentos_file:
                logger.warning("No se encontró el archivo fragmentos_metadata.json")
                self.fragmentos_originales = []
                return
            
            logger.info(f"Cargando fragmentos originales desde: {fragmentos_file}")
            with open(fragmentos_file, 'r', encoding='utf-8') as f:
                self.fragmentos_originales = json.load(f)
            
            logger.info(f"Cargados {len(self.fragmentos_originales)} fragmentos originales")
            
        except Exception as e:
            logger.error(f"Error cargando fragmentos originales: {e}")
            self.fragmentos_originales = []
    
    def obtener_texto_original(self, indice_global) -> str:
        """
        Obtiene el texto original de un fragmento usando su índice global.
        
        Args:
            indice_global: Índice del fragmento en la lista original (puede ser int o float)
            
        Returns:
            str: Texto original del fragmento o mensaje de error
        """
        try:
            # Convertir a entero si es necesario
            if isinstance(indice_global, float):
                indice_global = int(indice_global)
            elif not isinstance(indice_global, int):
                return f"Error: Tipo de índice inválido: {type(indice_global)}"
            
            if 0 <= indice_global < len(self.fragmentos_originales):
                return self.fragmentos_originales[indice_global].get("texto", "Texto no disponible")
            else:
                return f"Error: Índice {indice_global} fuera de rango (0-{len(self.fragmentos_originales)-1})"
        except Exception as e:
            logger.error(f"Error obteniendo texto original: {e}")
            return "Error al recuperar texto original"
    
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
            # Obtener estadísticas del índice para saber cuántos vectores hay
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                logger.warning("No hay vectores en el índice")
                return []
            
            # Realizar una búsqueda con un vector vacío para obtener múltiples resultados
            # Usar un número alto para asegurar que obtenemos muestras de todos los libros
            sample_size = min(100, total_vectors)  # Tomar hasta 100 muestras
            
            results = self.index.query(
                vector=[0.0] * 1024,  # Vector vacío
                top_k=sample_size,
                include_metadata=True
            )
            
            # Obtener todos los libros únicos
            libros = set()
            for match in results["matches"]:
                libro = match["metadata"].get("libro")
                if libro:
                    libros.add(libro)
            
            logger.info(f"Encontrados {len(libros)} libros únicos en {sample_size} muestras")
            return sorted(list(libros))  # Ordenar alfabéticamente
            
        except Exception as e:
            logger.error(f"Error obteniendo libros disponibles: {e}")
            # Como fallback, intentar obtener libros desde el JSON local
            try:
                libros_local = set()
                for fragmento in self.fragmentos_originales:
                    libro = fragmento.get("libro")
                    if libro:
                        libros_local.add(libro)
                logger.info(f"Usando libros del JSON local: {len(libros_local)} libros")
                return sorted(list(libros_local))
            except Exception as e2:
                logger.error(f"Error obteniendo libros del JSON local: {e2}")
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
            
            # Obtener texto original usando el índice global
            indice_global = match['metadata'].get('indice_global', -1)
            if indice_global >= 0:
                texto_original = self.obtener_texto_original(indice_global)
                print(f"Texto: {texto_original}")
            else:
                print("Texto: No disponible (índice no encontrado)")
            
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
