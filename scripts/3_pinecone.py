"""
SISTEMA DE SUBIDA A PINECONE MEJORADO - ANA-CHATBOT
==================================================

Este script se encarga de subir embeddings mejorados a la base de datos vectorial Pinecone.
Trabaja con el nuevo formato de metadatos estructurados.

FUNCIONALIDADES:
- Carga embeddings desde archivo embeddings_mejorados.pkl
- Conecta con Pinecone usando credenciales seguras
- Sube embeddings en lotes para optimizar rendimiento
- Valida datos antes de la subida
- Proporciona estadísticas detalladas del proceso
- Incluye metadatos mejorados (fuente, sección, subsección)

REQUISITOS:
- Archivo embeddings_mejorados.pkl generado por 2_embeddings_mejorado.py
- Variables de entorno configuradas (.env):
  * PINECONE_API_KEY: Clave API de Pinecone
  * PINECONE_HOST: URL del host de Pinecone
  * PINECONE_INDEX_NAME: Nombre del índice (opcional, default: "ana")

USO:
    python scripts/3_pinecone_mejorado.py

FLUJO DE TRABAJO:
    1. Procesar documentos → 1_frag_mejorado.py
    2. Generar embeddings → 2_embeddings_mejorado.py
    3. Subir a Pinecone → 3_pinecone_mejorado.py (este script)
    4. Usar chatbot → 4_consulta.py

EJEMPLO DE SALIDA:
    🚀 Iniciando proceso de subida mejorada a Pinecone...
    📄 Cargados 250 fragmentos con embeddings mejorados
    📚 Fuentes incluidas:
       - Diapositivas - Sistema Muscular: 45 fragmentos
       - Diapositivas - Sistema Esquelético: 52 fragmentos
       - Complemento Anatomía Funcional Humana: 153 fragmentos
    🔢 Dimensión de embeddings: 1024
    ✅ Conectado al índice 'ana'
    📤 Subiendo 250 embeddings en lotes de 50...
    📦 Procesando lote 1/5 (50 items)...
    ✅ Lote 1 subido exitosamente
    🎉 Todos los embeddings subidos exitosamente

DEPENDENCIAS:
- pinecone: Para conexión con base de datos vectorial
- pickle: Para cargar embeddings serializados
- python-dotenv: Para cargar variables de entorno

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

import pickle
import os
import sys
from pinecone import Pinecone
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cargar_embeddings_mejorados(archivo: str = "embeddings_mejorados.pkl") -> Tuple[List[Dict], List] | Tuple[None, None]:
    """
    Carga los embeddings mejorados y metadatos desde un archivo pickle.
    
    Args:
        archivo (str): Ruta al archivo pickle con embeddings mejorados. Default: "embeddings_mejorados.pkl"
        
    Returns:
        Tuple[List[Dict], List] | Tuple[None, None]: 
            - Lista de fragmentos con metadatos y embeddings
            - None si hay errores
            
    Example:
        >>> fragmentos, embeddings = cargar_embeddings_mejorados("embeddings_mejorados.pkl")
        >>> if fragmentos:
        >>>     print(f"Cargados {len(fragmentos)} fragmentos mejorados")
    """
    try:
        if not os.path.exists(archivo):
            logger.error(f"❌ Error: El archivo '{archivo}' no existe")
            return None, None
            
        logger.info(f"📁 Intentando cargar archivo: {archivo}")
        logger.info(f"📏 Tamaño del archivo: {os.path.getsize(archivo)} bytes")
        
        with open(archivo, "rb") as f:
            data = pickle.load(f)
            
        logger.info(f"✅ Archivo pickle cargado correctamente")
        logger.info(f"📋 Claves disponibles: {list(data.keys())}")
            
        if "fragmentos" not in data or "embeddings" not in data:
            logger.error("❌ Error: Formato de archivo inválido")
            logger.info(f"💡 Claves esperadas: 'fragmentos', 'embeddings'")
            logger.info(f"💡 Claves encontradas: {list(data.keys())}")
            return None, None
            
        fragmentos = data["fragmentos"]
        embeddings = data["embeddings"]
        
        logger.info(f"📄 Cargados {len(fragmentos)} fragmentos con embeddings mejorados")
        
        # Mostrar información de fuentes
        if "estadisticas_fuentes" in data:
            logger.info(f"📚 Fuentes incluidas:")
            for fuente, stats in data["estadisticas_fuentes"].items():
                logger.info(f"   - {fuente}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras")
        
        # Mostrar información de secciones
        if "estadisticas_secciones" in data:
            logger.info(f"📖 Secciones incluidas:")
            for seccion, stats in data["estadisticas_secciones"].items():
                logger.info(f"   - {seccion}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras")
        
        # Verificar dimensión de embeddings
        if embeddings is not None and len(embeddings) > 0:
            try:
                # Obtener la dimensión del primer embedding (array de NumPy)
                dimension = embeddings[0].shape[0] if hasattr(embeddings[0], 'shape') else len(embeddings[0])
                logger.info(f"🔢 Dimensión de embeddings: {dimension}")
                
                # Verificar si la dimensión es compatible con Pinecone (debe ser 1024 para tu índice)
                if dimension != 1024:
                    logger.warning(f"⚠️ Advertencia: Los embeddings tienen dimensión {dimension}, pero el índice 'ana' espera 1024")
                    logger.info("💡 Considera regenerar los embeddings con el modelo correcto")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo verificar la dimensión de embeddings: {e}")
        
        return fragmentos, embeddings
        
    except Exception as e:
        logger.error(f"❌ Error cargando embeddings mejorados: {e}")
        return None, None

def conectar_pinecone(api_key: str, index_name: str):
    """
    Conecta a Pinecone serverless usando el SDK v3 (pinecone-client >= 6.x).
    """
    try:
        pc = Pinecone(api_key=api_key)

        # Conectar al índice
        index = pc.Index(index_name)

        stats = index.describe_index_stats()
        logger.info(f"✅ Conectado al índice '{index_name}'")
        logger.info(f"📊 Estadísticas del índice: {stats}")
        return index

    except Exception as e:
        logger.error(f"❌ Error conectando a Pinecone: {e}")
        return None

def preparar_datos_mejorados(fragmentos: List[Dict], embeddings: List) -> List[Dict]:
    """
    Prepara los datos mejorados para subir a Pinecone en el formato requerido.
    
    Args:
        fragmentos (List[Dict]): Lista de fragmentos con metadatos mejorados
        embeddings (List): Lista de embeddings correspondientes
        
    Returns:
        List[Dict]: Lista de items formateados para Pinecone
        
    Example:
        >>> items = preparar_datos_mejorados(fragmentos, embeddings)
        >>> print(f"Preparados {len(items)} items para subir")
    """
    items = []
    
    for i, emb in enumerate(embeddings):
        try:
            fragmento = fragmentos[i]
            id_fragmento = fragmento["id"]
            fuente = fragmento.get("source", "desconocido")
            seccion = fragmento.get("section", "general")
            subseccion = fragmento.get("subsection", "general")
            
            # Metadatos mejorados incluyendo toda la información estructurada
            metadata = {
                "fragmento_id": id_fragmento,
                "source": fuente,
                "section": seccion,
                "subsection": subseccion,
                "page_number": fragmento.get("page_number", 0),
                "word_count": fragmento.get("metadata", {}).get("word_count", 0),
                "chunk_index": fragmento.get("metadata", {}).get("chunk_index", 0),
                "total_chunks": fragmento.get("metadata", {}).get("total_chunks", 0),
                "file_name": fragmento.get("metadata", {}).get("file_name", ""),
                "indice_global": i  # Índice para buscar en el JSON original
            }
            
            items.append({
                "id": id_fragmento,
                "values": emb.tolist(),
                "metadata": metadata
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Error procesando fragmento {i}: {e}")
            continue
    
    return items

def subir_embeddings_mejorados(index, items: List[Dict], batch_size: int = 50) -> bool:
    """
    Sube los embeddings mejorados a Pinecone en lotes para optimizar rendimiento.
    
    Args:
        index (Pinecone.Index): Índice de Pinecone
        items (List[Dict]): Lista de items a subir
        batch_size (int): Tamaño de lote para subida. Default: 50
        
    Returns:
        bool: True si la subida fue exitosa, False si hubo errores
        
    Example:
        >>> success = subir_embeddings_mejorados(index, items, batch_size=50)
        >>> if success:
        >>>     print("Subida completada")
    """
    try:
        total_items = len(items)
        logger.info(f"📤 Subiendo {total_items} embeddings mejorados en lotes de {batch_size}...")
        
        # Estadísticas por fuente
        fuentes_stats = {}
        secciones_stats = {}
        
        for item in items:
            fuente = item["metadata"].get("source", "desconocido")
            seccion = item["metadata"].get("section", "general")
            
            if fuente not in fuentes_stats:
                fuentes_stats[fuente] = 0
            fuentes_stats[fuente] += 1
            
            if seccion not in secciones_stats:
                secciones_stats[seccion] = 0
            secciones_stats[seccion] += 1
        
        logger.info(f"📚 Distribución por fuente:")
        for fuente, count in fuentes_stats.items():
            logger.info(f"   - {fuente}: {count} fragmentos")
        
        logger.info(f"📖 Distribución por sección:")
        for seccion, count in secciones_stats.items():
            logger.info(f"   - {seccion}: {count} fragmentos")
        
        for i in range(0, total_items, batch_size):
            batch = items[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_items + batch_size - 1) // batch_size
            
            logger.info(f"📦 Procesando lote {batch_num}/{total_batches} ({len(batch)} items)...")
            
            try:
                index.upsert(vectors=batch)
                logger.info(f"✅ Lote {batch_num} subido exitosamente")
            except Exception as e:
                logger.error(f"❌ Error subiendo lote {batch_num}: {e}")
                return False
        
        logger.info(f"🎉 Todos los embeddings mejorados subidos exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso de subida: {e}")
        return False

def main():
    """
    Función principal que ejecuta el proceso completo de subida mejorada a Pinecone.
    
    Esta función:
    1. Carga embeddings mejorados desde archivo pickle
    2. Valida credenciales de Pinecone
    3. Conecta con el índice de Pinecone
    4. Prepara datos en formato requerido con metadatos mejorados
    5. Sube embeddings en lotes
    6. Proporciona estadísticas detalladas del proceso
    
    Returns:
        bool: True si el proceso fue exitoso, False si hubo errores
    """
    try:
        # Cargar variables de entorno desde .env
        load_dotenv()
        
        # Configuración - Usar variables de entorno
        API_KEY = os.getenv("PINECONE_API_KEY")
        HOST = os.getenv("PINECONE_HOST")
        INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ana")
        BATCH_SIZE = 50
        
        # Validar que las credenciales estén configuradas
        if not API_KEY or not HOST:
            logger.error("❌ Error: Faltan variables de entorno")
            logger.info("💡 Configura PINECONE_API_KEY y PINECONE_HOST en tu archivo .env")
            return False
        

        logger.info("🚀 Iniciando proceso de subida mejorada a Pinecone...")
        
        # 1. Cargar embeddings mejorados
        fragmentos, embeddings = cargar_embeddings_mejorados()
        if fragmentos is None or embeddings is None:
            return False
            
        # 2. Conectar a Pinecone
        index = conectar_pinecone(API_KEY, INDEX_NAME)
        if index is None:
            return False
            
        # 3. Preparar datos mejorados
        items = preparar_datos_mejorados(fragmentos, embeddings)
        if not items:
            logger.error("❌ No hay datos válidos para subir")
            return False
            
        # 4. Subir embeddings mejorados
        success = subir_embeddings_mejorados(index, items, BATCH_SIZE)
        
        if success:
            logger.info("✅ Proceso completado exitosamente")
            logger.info(f"📊 Resumen: {len(items)} embeddings mejorados subidos a Pinecone")
            
            # Estadísticas finales
            fuentes_unicas = set(item["metadata"]["source"] for item in items)
            secciones_unicas = set(item["metadata"]["section"] for item in items)
            total_palabras = sum(item["metadata"]["word_count"] for item in items)
            
            logger.info(f"📈 Estadísticas finales:")
            logger.info(f"   - Fuentes únicas: {len(fuentes_unicas)}")
            logger.info(f"   - Secciones únicas: {len(secciones_unicas)}")
            logger.info(f"   - Total de palabras: {total_palabras}")
            logger.info(f"   - Promedio de palabras por fragmento: {total_palabras / len(items):.1f}")
        else:
            logger.error("❌ El proceso falló")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 