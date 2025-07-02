"""
GENERADOR DE EMBEDDINGS - ANA-CHATBOT
====================================

Este script genera embeddings semánticos a partir de fragmentos de texto de documentos
de anatomía. Es el segundo paso del pipeline de procesamiento.

FUNCIONALIDADES:
- Carga fragmentos de texto desde metadatos JSON
- Genera embeddings usando modelo BAAI/bge-large-en-v1.5
- Valida fragmentos antes del procesamiento
- Proporciona estadísticas detalladas por libro
- Guarda embeddings en formato pickle con metadatos

REQUISITOS:
- Archivo fragmentos_metadata.json generado por frag.py
- GPU recomendada para mejor rendimiento (opcional)

USO:
    python scripts/embeddings.py

FLUJO DE TRABAJO:
    1. Procesar documentos → frag.py (genera fragmentos_metadata.json)
    2. Generar embeddings → embeddings.py (este script)
    3. Subir a Pinecone → pinecone_u.py
    4. Usar chatbot → consulta.py

EJEMPLO DE SALIDA:
    ✅ GPU detectada, se usará para crear embeddings.
    🔧 GPU: NVIDIA GeForce RTX 3080
    🔄 Cargando modelo de embeddings...
    📁 Cargando metadatos de fragmentos...
    📄 Encontrados 150 fragmentos para procesar
    📚 Fragmentos válidos por libro:
       - G_A_S_4_E: 75 fragmentos, 37500 palabras
       - LIBRO_IFSSA: 75 fragmentos, 37500 palabras
    🧠 Generando embeddings...
    💾 Guardando embeddings...
    ✅ Embeddings guardados en embeddings.pkl
    📊 Total de fragmentos procesados: 150
    🔢 Dimensiones de embeddings: (150, 1024)

DEPENDENCIAS:
- sentence-transformers: Para generar embeddings semánticos
- torch: Framework de machine learning
- pickle: Para serializar embeddings
- json: Para cargar metadatos

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

from sentence_transformers import SentenceTransformer
import os
import pickle
import sys
import torch
import json

def crear_embeddings():
    """
    Función principal que genera embeddings semánticos para todos los fragmentos.
    
    Esta función:
    1. Verifica disponibilidad de GPU
    2. Carga el modelo de embeddings
    3. Valida archivos de entrada
    4. Procesa fragmentos de texto desde JSON
    5. Genera embeddings semánticos
    6. Guarda resultados con metadatos
    
    Returns:
        bool: True si el proceso fue exitoso, False si hubo errores
        
    Example:
        >>> success = crear_embeddings()
        >>> if success:
        >>>     print("Embeddings generados exitosamente")
    """
    try:
        # Obtener la ruta del directorio raíz del proyecto
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Verificar GPU
        if not torch.cuda.is_available():
            print("⚠️ Advertencia: No se detectó GPU, se usará CPU. Esto será más lento.")
        else:
            print("✅ GPU detectada, se usará para crear embeddings.")
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        
        # Cargar modelo
        print("🔄 Cargando modelo de embeddings...")
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        # Verificar que existe el archivo de metadatos
        metadata_file = os.path.join(project_root, "data", "fragmentos_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"❌ Error: El archivo '{metadata_file}' no existe")
            print("💡 Ejecuta primero el script frag.py para generar los fragmentos")
            return False
        
        # Cargar metadatos
        print("📁 Cargando metadatos de fragmentos...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            fragmentos = json.load(f)
        
        if not fragmentos:
            print(f"❌ Error: No se encontraron fragmentos en '{metadata_file}'")
            return False
            
        print(f"📄 Encontrados {len(fragmentos)} fragmentos para procesar")
        
        # Validar que los fragmentos tienen el texto completo
        fragmentos_validos = []
        for fragmento in fragmentos:
            if "texto" in fragmento and fragmento["texto"].strip():
                fragmentos_validos.append(fragmento)
            else:
                print(f"⚠️ Fragmento {fragmento.get('id', 'desconocido')} no tiene texto válido")
        
        if not fragmentos_validos:
            print("❌ Error: No se encontraron fragmentos con texto válido")
            return False
        
        # Estadísticas por libro
        libros_stats = {}
        for fragmento in fragmentos_validos:
            libro = fragmento["libro"]
            if libro not in libros_stats:
                libros_stats[libro] = {"fragmentos": 0, "palabras": 0}
            libros_stats[libro]["fragmentos"] += 1
            libros_stats[libro]["palabras"] += fragmento["palabras"]
        
        print(f"\n📚 Fragmentos válidos por libro:")
        for libro, stats in libros_stats.items():
            print(f"   - {libro}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras")
        
        # Generar embeddings usando texto del JSON
        print("\n🧠 Generando embeddings...")
        texts = [f["texto"] for f in fragmentos_validos]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Guardar embeddings y metadatos
        print("💾 Guardando embeddings...")
        with open("embeddings.pkl", "wb") as f:
            pickle.dump({
                "fragmentos": fragmentos_validos, 
                "embeddings": embeddings,
                "modelo": "BAAI/bge-large-en-v1.5",
                "total_fragmentos": len(fragmentos_validos),
                "libros": list(libros_stats.keys()),
                "estadisticas_libros": libros_stats
            }, f)
        
        print(f"✅ Embeddings guardados en embeddings.pkl")
        print(f"📊 Total de fragmentos procesados: {len(fragmentos_validos)}")
        print(f"🔢 Dimensiones de embeddings: {embeddings.shape}")
        print(f"📚 Libros incluidos: {', '.join(libros_stats.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    crear_embeddings()
