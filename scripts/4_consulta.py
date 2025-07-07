"""
CHATBOT DE ANATOMÍA MEJORADO - SISTEMA DE BÚSQUEDA SEMÁNTICA
===========================================================

Este script implementa un chatbot especializado en anatomía mejorado que utiliza 
búsqueda semántica para responder consultas sobre contenido médico-anatómico 
almacenado en Pinecone con metadatos estructurados.

FUNCIONALIDADES:
- Búsqueda semántica en documentos de anatomía
- Filtrado por fuente específica (diapositivas vs manual)
- Filtrado por sección anatómica
- Interfaz interactiva de consultas mejorada
- Visualización de resultados con metadatos estructurados
- Soporte para múltiples fuentes y secciones

REQUISITOS:
- Variables de entorno configuradas (.env):
  * PINECONE_API_KEY: Clave API de Pinecone
  * PINECONE_HOST: URL del host de Pinecone
  * PINECONE_INDEX_NAME: Nombre del índice (opcional, default: "ana")

USO:
    python scripts/4_consulta_mejorado.py

EJEMPLO DE USO:
    $ python scripts/4_consulta_mejorado.py
    📚 Fuentes disponibles: Diapositivas - Sistema Muscular, Diapositivas - Sistema Esquelético, Complemento Anatomía Funcional Humana
    📖 Secciones disponibles: Sistema Muscular, Sistema Esquelético, Dorso, Miembro Superior
    🔍 CHATBOT DE ANATOMÍA MEJORADO
    ============================================================
    Ingrese su consulta: ¿Qué es el sistema nervioso central?
    
    Opciones de filtro:
    1. Buscar en todas las fuentes
    2. Buscar solo en diapositivas
    3. Buscar solo en manual
    4. Filtrar por sección específica
    
    --- Resultado 1 ---
    Puntuación: 0.892
    Fuente: Complemento Anatomía Funcional Humana
    Sección: Sistema Nervioso
    Subsección: Sistema Nervioso Central
    Página: 45
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
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class SemanticSearchMejorado:
    """
    Clase para realizar búsquedas semánticas mejoradas en documentos de anatomía.
    
    Esta clase maneja la conexión con Pinecone, la generación de embeddings
    y la búsqueda semántica de contenido médico con metadatos estructurados.
    
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
        Inicializa el sistema de búsqueda semántica mejorado.
        
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
            self.model = SentenceTransformer("BAAI/bge-m3")
            
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
            
            # Cargar fragmentos originales desde JSON mejorado
            self.cargar_fragmentos_originales()
            
        except Exception as e:
            logger.error(f"Error al inicializar el sistema: {e}")
            raise
    
    def cargar_fragmentos_originales(self):
        """
        Carga los fragmentos originales desde el archivo JSON mejorado para recuperar el texto completo.
        """
        try:
            # Buscar el archivo JSON mejorado en diferentes ubicaciones posibles
            posibles_rutas = [
                "data/fragmentos_mejorados.json",
                "../data/fragmentos_mejorados.json",
                "scripts/../data/fragmentos_mejorados.json"
            ]
            
            fragmentos_file = None
            for ruta in posibles_rutas:
                if os.path.exists(ruta):
                    fragmentos_file = ruta
                    break
            
            if not fragmentos_file:
                logger.warning("No se encontró el archivo fragmentos_mejorados.json")
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
                return self.fragmentos_originales[indice_global].get("text", "Texto no disponible")
            else:
                return f"Error: Índice {indice_global} fuera de rango (0-{len(self.fragmentos_originales)-1})"
        except Exception as e:
            logger.error(f"Error obteniendo texto original: {e}")
            return "Error al recuperar texto original"
    
    def search(self, query: str, top_k: int = 5, fuente_filtro: Optional[str] = None, 
               seccion_filtro: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda semántica mejorada en el índice de Pinecone.
        
        Args:
            query (str): Texto de la consulta a buscar
            top_k (int, opcional): Número máximo de resultados a retornar. Default: 5
            fuente_filtro (str, opcional): Filtrar resultados por fuente específica. Default: None
            seccion_filtro (str, opcional): Filtrar resultados por sección específica. Default: None
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados con scores y metadatos
            
        Raises:
            Exception: Si hay errores en la búsqueda
            
        Example:
            >>> searcher = SemanticSearchMejorado()
            >>> results = searcher.search("sistema nervioso", top_k=3, fuente_filtro="Complemento Anatomía Funcional Humana")
            >>> print(f"Encontrados {len(results)} resultados")
        """
        try:
            logger.info(f"Buscando: '{query}'")
            if fuente_filtro:
                logger.info(f"Filtro de fuente: '{fuente_filtro}'")
            if seccion_filtro:
                logger.info(f"Filtro de sección: '{seccion_filtro}'")
            
            # Codificar la consulta
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Realizar búsqueda con más resultados si hay filtros
            search_k = top_k * 3 if (fuente_filtro or seccion_filtro) else top_k
            
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=search_k,
                include_metadata=True
            )
            
            # Aplicar filtros si se especifican
            filtered_results = []
            for match in results["matches"]:
                metadata = match["metadata"]
                
                # Filtrar por fuente
                if fuente_filtro and metadata.get("source") != fuente_filtro:
                    continue
                
                # Filtrar por sección
                if seccion_filtro and metadata.get("section") != seccion_filtro:
                    continue
                
                filtered_results.append(match)
                if len(filtered_results) >= top_k:
                    break
            
            logger.info(f"Encontrados {len(filtered_results)} resultados después de filtros")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error en la búsqueda: {e}")
            raise
    
    def get_available_sources(self) -> List[str]:
        """
        Obtiene la lista de fuentes disponibles en el índice.
        
        Returns:
            List[str]: Lista de nombres de fuentes únicos
            
        Example:
            >>> searcher = SemanticSearchMejorado()
            >>> fuentes = searcher.get_available_sources()
            >>> print(f"Fuentes disponibles: {fuentes}")
        """
        try:
            # Obtener estadísticas del índice
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                logger.warning("No hay vectores en el índice")
                return []
            
            # Realizar una búsqueda con un vector vacío para obtener múltiples resultados
            sample_size = min(100, total_vectors)
            
            results = self.index.query(
                vector=[0.0] * 1024,  # Vector vacío
                top_k=sample_size,
                include_metadata=True
            )
            
            # Obtener todas las fuentes únicas
            fuentes = set()
            for match in results["matches"]:
                fuente = match["metadata"].get("source")
                if fuente:
                    fuentes.add(fuente)
            
            logger.info(f"Encontradas {len(fuentes)} fuentes únicas en {sample_size} muestras")
            return sorted(list(fuentes))
            
        except Exception as e:
            logger.error(f"Error obteniendo fuentes disponibles: {e}")
            # Como fallback, intentar obtener fuentes desde el JSON local
            try:
                fuentes_local = set()
                for fragmento in self.fragmentos_originales:
                    fuente = fragmento.get("source")
                    if fuente:
                        fuentes_local.add(fuente)
                logger.info(f"Usando fuentes del JSON local: {len(fuentes_local)} fuentes")
                return sorted(list(fuentes_local))
            except Exception as e2:
                logger.error(f"Error obteniendo fuentes del JSON local: {e2}")
                return []
    
    def get_available_sections(self) -> List[str]:
        """
        Obtiene la lista de secciones disponibles en el índice.
        
        Returns:
            List[str]: Lista de nombres de secciones únicos
        """
        try:
            # Obtener estadísticas del índice
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                return []
            
            # Realizar una búsqueda con un vector vacío para obtener múltiples resultados
            sample_size = min(100, total_vectors)
            
            results = self.index.query(
                vector=[0.0] * 1024,  # Vector vacío
                top_k=sample_size,
                include_metadata=True
            )
            
            # Obtener todas las secciones únicas
            secciones = set()
            for match in results["matches"]:
                seccion = match["metadata"].get("section")
                if seccion:
                    secciones.add(seccion)
            
            logger.info(f"Encontradas {len(secciones)} secciones únicas en {sample_size} muestras")
            return sorted(list(secciones))
            
        except Exception as e:
            logger.error(f"Error obteniendo secciones disponibles: {e}")
            # Como fallback, intentar obtener secciones desde el JSON local
            try:
                secciones_local = set()
                for fragmento in self.fragmentos_originales:
                    seccion = fragmento.get("section")
                    if seccion:
                        secciones_local.add(seccion)
                logger.info(f"Usando secciones del JSON local: {len(secciones_local)} secciones")
                return sorted(list(secciones_local))
            except Exception as e2:
                logger.error(f"Error obteniendo secciones del JSON local: {e2}")
                return []
    
    def display_results_mejorados(self, matches: List[Dict[str, Any]]):
        """
        Muestra los resultados de búsqueda mejorados de forma legible en consola.
        
        Args:
            matches (List[Dict[str, Any]]): Lista de resultados de búsqueda
            
        Example:
            >>> results = searcher.search("anatomía")
            >>> searcher.display_results_mejorados(results)
        """
        if not matches:
            print("No se encontraron resultados.")
            return
            
        for i, match in enumerate(matches, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Puntuación: {match['score']:.3f}")
            print(f"Fuente: {match['metadata'].get('source', 'Desconocido')}")
            print(f"Sección: {match['metadata'].get('section', 'N/A')}")
            print(f"Subsección: {match['metadata'].get('subsection', 'N/A')}")
            print(f"Página: {match['metadata'].get('page_number', 'N/A')}")
            print(f"Palabras: {match['metadata'].get('word_count', 0)}")
            
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
    Función principal que ejecuta el chatbot interactivo mejorado.
    
    Esta función:
    1. Inicializa el sistema de búsqueda mejorado
    2. Muestra las fuentes y secciones disponibles
    3. Ejecuta un bucle interactivo para consultas
    4. Permite filtros avanzados por fuente y sección
    5. Muestra resultados con metadatos estructurados
    
    Para salir del programa, escribe 'salir' cuando se solicite la consulta.
    """
    try:
        # Crear instancia del buscador mejorado
        searcher = SemanticSearchMejorado()
        
        # Mostrar fuentes y secciones disponibles
        fuentes_disponibles = searcher.get_available_sources()
        secciones_disponibles = searcher.get_available_sections()
        
        if fuentes_disponibles:
            print(f"📚 Fuentes disponibles: {', '.join(fuentes_disponibles)}")
        else:
            print("⚠️ No se pudieron obtener las fuentes disponibles")
        
        if secciones_disponibles:
            print(f"📖 Secciones disponibles: {', '.join(secciones_disponibles)}")
        else:
            print("⚠️ No se pudieron obtener las secciones disponibles")
        
        while True:
            print("\n" + "="*60)
            print("🔍 CHATBOT DE ANATOMÍA MEJORADO")
            print("="*60)
            
            # Consulta de ejemplo (puede ser interactiva)
            query = input("Ingrese su consulta: ")
            if query.lower() == "salir":
                break
            
            # Opciones de filtro mejoradas
            print("\nOpciones de filtro:")
            print("1. Buscar en todas las fuentes")
            print("2. Buscar solo en diapositivas")
            print("3. Buscar solo en manual")
            print("4. Filtrar por fuente específica")
            print("5. Filtrar por sección específica")
            
            try:
                opcion = input(f"\nSeleccione opción (1-5): ")
                opcion = int(opcion)
                
                fuente_filtro = None
                seccion_filtro = None
                
                if opcion == 1:
                    # Buscar en todas las fuentes
                    pass
                elif opcion == 2:
                    # Buscar solo en diapositivas
                    fuente_filtro = "Diapositivas"
                elif opcion == 3:
                    # Buscar solo en manual
                    fuente_filtro = "Complemento Anatomía Funcional Humana"
                elif opcion == 4:
                    # Filtrar por fuente específica
                    print("\nFuentes disponibles:")
                    for i, fuente in enumerate(fuentes_disponibles, 1):
                        print(f"{i}. {fuente}")
                    try:
                        fuente_idx = int(input("Seleccione fuente: ")) - 1
                        if 0 <= fuente_idx < len(fuentes_disponibles):
                            fuente_filtro = fuentes_disponibles[fuente_idx]
                    except ValueError:
                        print("Opción inválida")
                elif opcion == 5:
                    # Filtrar por sección específica
                    print("\nSecciones disponibles:")
                    for i, seccion in enumerate(secciones_disponibles, 1):
                        print(f"{i}. {seccion}")
                    try:
                        seccion_idx = int(input("Seleccione sección: ")) - 1
                        if 0 <= seccion_idx < len(secciones_disponibles):
                            seccion_filtro = secciones_disponibles[seccion_idx]
                    except ValueError:
                        print("Opción inválida")
                else:
                    print("Opción inválida, buscando en todas las fuentes...")
                    
            except ValueError:
                print("Opción inválida, buscando en todas las fuentes...")
            
            # Realizar búsqueda
            results = searcher.search(query, top_k=3, fuente_filtro=fuente_filtro, seccion_filtro=seccion_filtro)
        
            # Mostrar resultados
            searcher.display_results_mejorados(results)
        
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 