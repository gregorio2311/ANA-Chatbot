"""
GENERADOR DE EMBEDDINGS MEJORADO - ANA-CHATBOT
=============================================

Este script genera embeddings semánticos a partir de fragmentos mejorados
de documentos de anatomía. Trabaja con el nuevo formato JSON estructurado.

FUNCIONALIDADES:
- Carga fragmentos desde fragmentos_mejorados.json
- Genera embeddings usando modelo BAAI/bge-m3
- Valida fragmentos antes del procesamiento
- Proporciona estadísticas detalladas por fuente y sección
- Guarda embeddings en formato pickle con metadatos mejorados

REQUISITOS:
- Archivo fragmentos_mejorados.json generado por 1_frag_mejorado.py
- GPU recomendada para mejor rendimiento (opcional)

USO:
    python scripts/2_embeddings_mejorado.py

FLUJO DE TRABAJO:
    1. Procesar documentos → 1_frag_mejorado.py (genera fragmentos_mejorados.json)
    2. Generar embeddings → 2_embeddings_mejorado.py (este script)
    3. Subir a Pinecone → 3_pinecone_u.py
    4. Usar chatbot → 4_consulta.py

EJEMPLO DE SALIDA:
    ✅ GPU detectada, se usará para crear embeddings.
    🔧 GPU: NVIDIA GeForce RTX 3080
    🔄 Cargando modelo de embeddings...
    📁 Cargando metadatos de fragmentos mejorados...
    📄 Encontrados 250 fragmentos para procesar
    📚 Fragmentos válidos por fuente:
       - Diapositivas - Sistema Muscular: 45 fragmentos, 22500 palabras
       - Diapositivas - Sistema Esquelético: 52 fragmentos, 26000 palabras
       - Complemento Anatomía Funcional Humana: 153 fragmentos, 76500 palabras
    🧠 Generando embeddings...
    💾 Guardando embeddings...
    ✅ Embeddings guardados en embeddings_mejorados.pkl
    📊 Total de fragmentos procesados: 250
    🔢 Dimensiones de embeddings: (250, 1024)

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
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crear_embeddings_mejorados():
    """
    Función principal que genera embeddings semánticos para todos los fragmentos mejorados.
    
    Esta función:
    1. Verifica disponibilidad de GPU
    2. Carga el modelo de embeddings
    3. Valida archivos de entrada
    4. Procesa fragmentos de texto desde JSON mejorado
    5. Genera embeddings semánticos
    6. Guarda resultados con metadatos mejorados
    
    Returns:
        bool: True si el proceso fue exitoso, False si hubo errores
        
    Example:
        >>> success = crear_embeddings_mejorados()
        >>> if success:
        >>>     print("Embeddings mejorados generados exitosamente")
    """
    try:
        # Obtener la ruta del directorio raíz del proyecto
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Verificar GPU
        if not torch.cuda.is_available():
            logger.warning("⚠️ Advertencia: No se detectó GPU, se usará CPU. Esto será más lento.")
        else:
            logger.info("✅ GPU detectada, se usará para crear embeddings.")
            logger.info(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        
        # Cargar modelo
        logger.info("🔄 Cargando modelo de embeddings...")
        model = SentenceTransformer("BAAI/bge-m3")
        
        # Verificar que existe el archivo de metadatos mejorados
        metadata_file = os.path.join(project_root, "data", "fragmentos_mejorados.json")
        if not os.path.exists(metadata_file):
            logger.error(f"❌ Error: El archivo '{metadata_file}' no existe")
            logger.info("💡 Ejecuta primero el script 1_frag_mejorado.py para generar los fragmentos")
            return False
        
        # Cargar metadatos mejorados
        logger.info("📁 Cargando metadatos de fragmentos mejorados...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            fragmentos = json.load(f)
        
        if not fragmentos:
            logger.error(f"❌ Error: No se encontraron fragmentos en '{metadata_file}'")
            return False
            
        logger.info(f"📄 Encontrados {len(fragmentos)} fragmentos para procesar")
        
        # Validar que los fragmentos tienen el texto completo
        fragmentos_validos = []
        for fragmento in fragmentos:
            if "text" in fragmento and fragmento["text"].strip():
                fragmentos_validos.append(fragmento)
            else:
                logger.warning(f"⚠️ Fragmento {fragmento.get('id', 'desconocido')} no tiene texto válido")
        
        if not fragmentos_validos:
            logger.error("❌ Error: No se encontraron fragmentos con texto válido")
            return False
        
        # Estadísticas por fuente
        fuentes_stats = {}
        secciones_stats = {}
        
        for fragmento in fragmentos_validos:
            fuente = fragmento["source"]
            seccion = fragmento["section"]
            palabras = fragmento["metadata"]["word_count"]
            
            # Estadísticas por fuente
            if fuente not in fuentes_stats:
                fuentes_stats[fuente] = {"fragmentos": 0, "palabras": 0, "secciones": set()}
            fuentes_stats[fuente]["fragmentos"] += 1
            fuentes_stats[fuente]["palabras"] += palabras
            fuentes_stats[fuente]["secciones"].add(seccion)
            
            # Estadísticas por sección
            if seccion not in secciones_stats:
                secciones_stats[seccion] = {"fragmentos": 0, "palabras": 0, "fuentes": set()}
            secciones_stats[seccion]["fragmentos"] += 1
            secciones_stats[seccion]["palabras"] += palabras
            secciones_stats[seccion]["fuentes"].add(fuente)
        
        logger.info(f"\n📚 Fragmentos válidos por fuente:")
        for fuente, stats in fuentes_stats.items():
            logger.info(f"   - {fuente}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras, {len(stats['secciones'])} secciones")
        
        logger.info(f"\n📖 Fragmentos válidos por sección:")
        for seccion, stats in secciones_stats.items():
            logger.info(f"   - {seccion}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras, {len(stats['fuentes'])} fuentes")
        
        # Generar embeddings usando texto del JSON
        logger.info("\n🧠 Generando embeddings...")
        texts = [f["text"] for f in fragmentos_validos]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Preparar metadatos mejorados para guardar
        metadatos_mejorados = {
            "fragmentos": fragmentos_validos, 
            "embeddings": embeddings,
            "modelo": "BAAI/bge-m3",
            "total_fragmentos": len(fragmentos_validos),
            "fuentes": list(fuentes_stats.keys()),
            "secciones": list(secciones_stats.keys()),
            "estadisticas_fuentes": {
                fuente: {
                    "fragmentos": stats["fragmentos"],
                    "palabras": stats["palabras"],
                    "secciones": list(stats["secciones"])
                } for fuente, stats in fuentes_stats.items()
            },
            "estadisticas_secciones": {
                seccion: {
                    "fragmentos": stats["fragmentos"],
                    "palabras": stats["palabras"],
                    "fuentes": list(stats["fuentes"])
                } for seccion, stats in secciones_stats.items()
            }
        }
        
        # Guardar embeddings y metadatos mejorados
        logger.info("💾 Guardando embeddings...")
        output_file = "embeddings_mejorados.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(metadatos_mejorados, f)
        
        logger.info(f"✅ Embeddings guardados en {output_file}")
        logger.info(f"📊 Total de fragmentos procesados: {len(fragmentos_validos)}")
        logger.info(f"🔢 Dimensiones de embeddings: {embeddings.shape}")
        logger.info(f"📚 Fuentes incluidas: {', '.join(fuentes_stats.keys())}")
        logger.info(f"📖 Secciones incluidas: {', '.join(secciones_stats.keys())}")
        
        # Mostrar estadísticas detalladas
        logger.info(f"\n📈 Estadísticas detalladas:")
        logger.info(f"   - Promedio de palabras por fragmento: {sum(f['metadata']['word_count'] for f in fragmentos_validos) / len(fragmentos_validos):.1f}")
        logger.info(f"   - Fragmento más largo: {max(f['metadata']['word_count'] for f in fragmentos_validos)} palabras")
        logger.info(f"   - Fragmento más corto: {min(f['metadata']['word_count'] for f in fragmentos_validos)} palabras")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = crear_embeddings_mejorados()
    if not success:
        sys.exit(1) 