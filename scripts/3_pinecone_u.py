"""
SISTEMA DE SUBIDA A PINECONE - ANA-CHATBOT
==========================================

Este script se encarga de subir embeddings generados a la base de datos vectorial Pinecone.
Es el paso final del pipeline de procesamiento de documentos de anatomía.

FUNCIONALIDADES:
- Carga embeddings desde archivo pickle
- Conecta con Pinecone usando credenciales seguras
- Sube embeddings en lotes para optimizar rendimiento
- Valida datos antes de la subida
- Proporciona estadísticas detalladas del proceso

REQUISITOS:
- Archivo embeddings.pkl generado por embeddings.py
- Variables de entorno configuradas (.env):
  * PINECONE_API_KEY: Clave API de Pinecone
  * PINECONE_HOST: URL del host de Pinecone
  * PINECONE_INDEX_NAME: Nombre del índice (opcional, default: "ana")

USO:
    python scripts/pinecone_u.py

FLUJO DE TRABAJO:
    1. Procesar documentos → frag.py
    2. Generar embeddings → embeddings.py
    3. Subir a Pinecone → pinecone_u.py (este script)
    4. Usar chatbot → consulta.py

EJEMPLO DE SALIDA:
    🚀 Iniciando proceso de subida a Pinecone...
    📄 Cargados 150 fragmentos con embeddings
    📚 Libros incluidos:
       - G_A_S_4_E: 75 fragmentos, 37500 palabras
       - LIBRO_IFSSA: 75 fragmentos, 37500 palabras
    🔢 Dimensión de embeddings: 1024
    ✅ Conectado al índice 'ana'
    📤 Subiendo 150 embeddings en lotes de 50...
    📦 Procesando lote 1/3 (50 items)...
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

def cargar_embeddings(archivo: str = "embeddings.pkl") -> Tuple[List[Dict], List] | Tuple[None, None]:
    """
    Carga los embeddings y metadatos desde un archivo pickle.
    
    Args:
        archivo (str): Ruta al archivo pickle con embeddings. Default: "embeddings.pkl"
        
    Returns:
        Tuple[List[Dict], List] | Tuple[None, None]: 
            - Lista de fragmentos con metadatos y embeddings
            - None si hay errores
            
    Example:
        >>> fragmentos, embeddings = cargar_embeddings("embeddings.pkl")
        >>> if fragmentos:
        >>>     print(f"Cargados {len(fragmentos)} fragmentos")
    """
    try:
        if not os.path.exists(archivo):
            print(f"❌ Error: El archivo '{archivo}' no existe")
            return None, None
            
        print(f"📁 Intentando cargar archivo: {archivo}")
        print(f"📏 Tamaño del archivo: {os.path.getsize(archivo)} bytes")
        
        with open(archivo, "rb") as f:
            data = pickle.load(f)
            
        print(f"✅ Archivo pickle cargado correctamente")
        print(f"📋 Claves disponibles: {list(data.keys())}")
            
        if "fragmentos" not in data or "embeddings" not in data:
            print("❌ Error: Formato de archivo inválido")
            print(f"💡 Claves esperadas: 'fragmentos', 'embeddings'")
            print(f"💡 Claves encontradas: {list(data.keys())}")
            return None, None
            
        fragmentos = data["fragmentos"]
        embeddings = data["embeddings"]
        
        print(f"📄 Cargados {len(fragmentos)} fragmentos con embeddings")
        
        # Mostrar información de libros
        if "estadisticas_libros" in data:
            print(f"📚 Libros incluidos:")
            for libro, stats in data["estadisticas_libros"].items():
                print(f"   - {libro}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras")
        
        # Verificar dimensión de embeddings
        if embeddings is not None and len(embeddings) > 0:
            try:
                # Obtener la dimensión del primer embedding (array de NumPy)
                dimension = embeddings[0].shape[0] if hasattr(embeddings[0], 'shape') else len(embeddings[0])
                print(f"🔢 Dimensión de embeddings: {dimension}")
                
                # Verificar si la dimensión es compatible con Pinecone (debe ser 1024 para tu índice)
                if dimension != 1024:
                    print(f"⚠️ Advertencia: Los embeddings tienen dimensión {dimension}, pero el índice 'ana' espera 1024")
                    print("💡 Considera regenerar los embeddings con el modelo correcto")
            except Exception as e:
                print(f"⚠️ No se pudo verificar la dimensión de embeddings: {e}")
        
        return fragmentos, embeddings
        
    except Exception as e:
        print(f"❌ Error cargando embeddings: {e}")
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
        print(f"✅ Conectado al índice '{index_name}'")
        print(f"📊 Estadísticas del índice: {stats}")
        return index

    except Exception as e:
        print(f"❌ Error conectando a Pinecone: {e}")
        return None



def preparar_datos(fragmentos: List[Dict], embeddings: List) -> List[Dict]:
    """
    Prepara los datos para subir a Pinecone en el formato requerido.
    
    Args:
        fragmentos (List[Dict]): Lista de fragmentos con metadatos
        embeddings (List): Lista de embeddings correspondientes
        
    Returns:
        List[Dict]: Lista de items formateados para Pinecone
        
    Example:
        >>> items = preparar_datos(fragmentos, embeddings)
        >>> print(f"Preparados {len(items)} items para subir")
    """
    items = []
    
    for i, emb in enumerate(embeddings):
        try:
            fragmento = fragmentos[i]
            id_fragmento = fragmento["id"]
            texto_original = fragmento["texto"]
            libro = fragmento.get("libro", "desconocido")
            
            texto_limpio = texto_original.strip()
            if not texto_limpio:
                print(f"⚠️ Fragmento {id_fragmento} está vacío, saltando...")
                continue
                
            metadata = {
                "texto": texto_limpio,
                "longitud": len(texto_limpio),
                "fragmento_id": id_fragmento,
                "libro": libro,
                "palabras": fragmento.get("palabras", 0),
                "indice": fragmento.get("indice", 0)
            }
            
            items.append({
                "id": id_fragmento,
                "values": emb.tolist(),
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"⚠️ Error procesando fragmento {i}: {e}")
            continue
    
    return items


def subir_embeddings(index: Pinecone.Index, items: List[Dict], batch_size: int = 50) -> bool:
    """
    Sube los embeddings a Pinecone en lotes para optimizar rendimiento.
    
    Args:
        index (Pinecone.Index): Índice de Pinecone
        items (List[Dict]): Lista de items a subir
        batch_size (int): Tamaño de lote para subida. Default: 50
        
    Returns:
        bool: True si la subida fue exitosa, False si hubo errores
        
    Example:
        >>> success = subir_embeddings(index, items, batch_size=50)
        >>> if success:
        >>>     print("Subida completada")
    """
    try:
        total_items = len(items)
        print(f"📤 Subiendo {total_items} embeddings en lotes de {batch_size}...")
        
        # Estadísticas por libro
        libros_stats = {}
        for item in items:
            libro = item["metadata"].get("libro", "desconocido")
            if libro not in libros_stats:
                libros_stats[libro] = 0
            libros_stats[libro] += 1
        
        print(f"📚 Distribución por libro:")
        for libro, count in libros_stats.items():
            print(f"   - {libro}: {count} fragmentos")
        
        for i in range(0, total_items, batch_size):
            batch = items[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_items + batch_size - 1) // batch_size
            
            print(f"📦 Procesando lote {batch_num}/{total_batches} ({len(batch)} items)...")
            
            try:
                index.upsert(vectors=batch)
                print(f"✅ Lote {batch_num} subido exitosamente")
            except Exception as e:
                print(f"❌ Error subiendo lote {batch_num}: {e}")
                return False
        
        print(f"🎉 Todos los embeddings subidos exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en el proceso de subida: {e}")
        return False

def main():
    """
    Función principal que ejecuta el proceso completo de subida a Pinecone.
    
    Esta función:
    1. Carga embeddings desde archivo pickle
    2. Valida credenciales de Pinecone
    3. Conecta con el índice de Pinecone
    4. Prepara datos en formato requerido
    5. Sube embeddings en lotes
    6. Proporciona estadísticas del proceso
    
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
            print("❌ Error: Faltan variables de entorno")
            print("💡 Configura PINECONE_API_KEY y PINECONE_HOST en tu archivo .env")
            return False
        

        print("🚀 Iniciando proceso de subida a Pinecone...")
        
        # 1. Cargar embeddings
        fragmentos, embeddings = cargar_embeddings()
        if fragmentos is None or embeddings is None:
            return False
            
        # 2. Conectar a Pinecone
        index = conectar_pinecone(API_KEY, INDEX_NAME)
        if index is None:
            return False
            
        # 3. Preparar datos
        items = preparar_datos(fragmentos, embeddings)
        if not items:
            print("❌ No hay datos válidos para subir")
            return False
            
        # 4. Subir embeddings
        success = subir_embeddings(index, items, BATCH_SIZE)
        
        if success:
            print("✅ Proceso completado exitosamente")
            print(f"📊 Resumen: {len(items)} embeddings subidos a Pinecone")
        else:
            print("❌ El proceso falló")
            
        return success
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    main()
