"""
PROCESADOR DE FRAGMENTOS - ANA-CHATBOT
=====================================

Este script procesa documentos Word (.docx) de anatomía y los divide en fragmentos
de texto optimizados para búsqueda semántica. Es el primer paso del pipeline.

FUNCIONALIDADES:
- Extrae texto de documentos Word (.docx)
- Divide texto en fragmentos inteligentes (~500 palabras)
- Preserva estructura semántica de párrafos
- Genera metadatos detallados por fragmento
- Crea archivo JSON con todos los fragmentos y metadatos

REQUISITOS:
- Documentos Word en carpeta data/libros_word/
- Configuración de libros en el script
- python-docx instalado

USO:
    python scripts/frag.py

FLUJO DE TRABAJO:
    1. Procesar documentos → frag.py (este script)
    2. Generar embeddings → embeddings.py
    3. Subir a Pinecone → pinecone_u.py
    4. Usar chatbot → consulta.py

CONFIGURACIÓN DE LIBROS:
    libros_config = [
        {
            "archivo": "data/libros_word/G_A_S_4_E.docx",
            "nombre": "G_A_S_4_E"
        },
        {
            "archivo": "data/libros_word/LIBRO_IFSSA.docx", 
            "nombre": "LIBRO_IFSSA"
        }
    ]

EJEMPLO DE SALIDA:
    🚀 Iniciando procesamiento de múltiples libros...
    📖 Cargando documento: G_A_S_4_E
    ✅ Texto extraído de G_A_S_4_E: 75000 palabras
    ✂️ Creando fragmentos...
    ✅ Se crearon 150 fragmentos para G_A_S_4_E
    🎉 Procesamiento completado exitosamente
    📊 Estadísticas generales:
       - Total de fragmentos: 300
       - Total de palabras: 150000
    📚 Estadísticas por libro:
       - G_A_S_4_E: 150 fragmentos, 75000 palabras
       - LIBRO_IFSSA: 150 fragmentos, 75000 palabras

DEPENDENCIAS:
- python-docx: Para procesar documentos Word
- re: Para limpieza de texto
- json: Para guardar metadatos

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

import os
from docx import Document
import re
import json

def limpiar_texto(texto):
    """
    Limpia el texto eliminando caracteres especiales y normalizando espacios.
    
    Args:
        texto (str): Texto original a limpiar
        
    Returns:
        str: Texto limpio y normalizado
        
    Example:
        >>> texto_sucio = "Hola   mundo!!!   "
        >>> texto_limpio = limpiar_texto(texto_sucio)
        >>> print(texto_limpio)  # "Hola mundo"
    """
    texto = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)áéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def dividir_texto_inteligente(texto, max_palabras=500):
    """
    Divide el texto en fragmentos inteligentes preservando la estructura semántica.
    
    Args:
        texto (str): Texto completo a dividir
        max_palabras (int): Número máximo de palabras por fragmento. Default: 500
        
    Returns:
        List[str]: Lista de fragmentos de texto
        
    Example:
        >>> texto_largo = "Párrafo 1... Párrafo 2... Párrafo 3..."
        >>> fragmentos = dividir_texto_inteligente(texto_largo, max_palabras=300)
        >>> print(f"Se crearon {len(fragmentos)} fragmentos")
    """
    texto = limpiar_texto(texto)
    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    
    fragmentos = []
    fragmento_actual = []
    palabras_actuales = 0
    
    for parrafo in parrafos:
        palabras_parrafo = parrafo.split()
        
        if len(palabras_parrafo) > max_palabras:
            oraciones = re.split(r'(?<=[.!?])\s+', parrafo)
            for oracion in oraciones:
                palabras_oracion = oracion.split()
                
                if palabras_actuales + len(palabras_oracion) <= max_palabras:
                    fragmento_actual.append(oracion)
                    palabras_actuales += len(palabras_oracion)
                else:
                    if fragmento_actual:
                        fragmentos.append(' '.join(fragmento_actual))
                    fragmento_actual = [oracion]
                    palabras_actuales = len(palabras_oracion)
        else:
            if palabras_actuales + len(palabras_parrafo) <= max_palabras:
                fragmento_actual.append(parrafo)
                palabras_actuales += len(palabras_parrafo)
            else:
                if fragmento_actual:
                    fragmentos.append(' '.join(fragmento_actual))
                fragmento_actual = [parrafo]
                palabras_actuales = len(palabras_parrafo)
    
    if fragmento_actual and palabras_actuales > 30:
        fragmentos.append(' '.join(fragmento_actual))
    
    return fragmentos

def procesar_libro(archivo_docx, nombre_libro, max_palabras=500):
    """
    Procesa un libro específico y retorna sus fragmentos con metadatos.
    
    Args:
        archivo_docx (str): Ruta al archivo Word (.docx)
        nombre_libro (str): Nombre identificador del libro
        max_palabras (int): Número máximo de palabras por fragmento. Default: 500
        
    Returns:
        List[Dict]: Lista de fragmentos con metadatos
        
    Example:
        >>> fragmentos = procesar_libro("data/libros_word/anatomia.docx", "ANATOMIA")
        >>> print(f"Procesados {len(fragmentos)} fragmentos")
    """
    if not os.path.exists(archivo_docx):
        print(f"❌ Error: No se encontró el archivo {archivo_docx}")
        return []
    
    try:
        print(f"📖 Cargando documento: {nombre_libro}")
        doc = Document(archivo_docx)
        texto_completo = [p.text for p in doc.paragraphs if p.text.strip()]
        texto = '\n'.join(texto_completo)
        
        if not texto.strip():
            print(f"❌ Error: No se pudo extraer texto del documento {nombre_libro}")
            return []
        
        print(f"✅ Texto extraído de {nombre_libro}: {len(texto.split())} palabras")
        
        print("✂️ Creando fragmentos...")
        fragmentos_texto = dividir_texto_inteligente(texto, max_palabras)
        
        # Crear fragmentos con metadatos
        fragmentos_con_metadata = []
        for idx, frag in enumerate(fragmentos_texto):
            palabras_frag = len(frag.split())
            fragmento_id = f"{nombre_libro}_fragmento_{idx+1:04d}"
            
            fragmento_data = {
                "id": fragmento_id,
                "texto": frag,
                "libro": nombre_libro,
                "palabras": palabras_frag,
                "indice": idx + 1
            }
            fragmentos_con_metadata.append(fragmento_data)
        
        print(f"✅ Se crearon {len(fragmentos_con_metadata)} fragmentos para {nombre_libro}")
        return fragmentos_con_metadata
        
    except Exception as e:
        print(f"❌ Error al procesar {nombre_libro}: {str(e)}")
        return []

def main():
    """
    Función principal que procesa múltiples libros de anatomía.
    
    Esta función:
    1. Define la configuración de libros a procesar
    2. Procesa cada libro individualmente
    3. Combina todos los fragmentos
    4. Genera archivo de metadatos JSON con texto completo
    5. Proporciona estadísticas detalladas
    
    Para agregar nuevos libros, modifica la lista libros_config.
    """
    # Obtener la ruta del directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuración de libros
    libros_config = [
        {
            "archivo": os.path.join(project_root, "data", "libros_word", "G_A_S_4_E.docx"),
            "nombre": "G_A_S_4_E"
        }
    ]
    
    max_palabras = 500
    todos_los_fragmentos = []
    
    print("🚀 Iniciando procesamiento de múltiples libros...")
    
    # Procesar cada libro
    for libro_config in libros_config:
        archivo = libro_config["archivo"]
        nombre = libro_config["nombre"]
        
        if os.path.exists(archivo):
            fragmentos = procesar_libro(archivo, nombre, max_palabras)
            todos_los_fragmentos.extend(fragmentos)
        else:
            print(f"⚠️ Archivo no encontrado: {archivo}")
    
    if not todos_los_fragmentos:
        print("❌ No se pudieron procesar fragmentos de ningún libro")
        return
    
    # Guardar metadatos completos con texto incluido
    metadata_file = os.path.join(project_root, "data", "fragmentos_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(todos_los_fragmentos, f, ensure_ascii=False, indent=2)
    
    # Estadísticas por libro
    libros_stats = {}
    for fragmento in todos_los_fragmentos:
        libro = fragmento["libro"]
        if libro not in libros_stats:
            libros_stats[libro] = {"fragmentos": 0, "palabras": 0}
        libros_stats[libro]["fragmentos"] += 1
        libros_stats[libro]["palabras"] += fragmento["palabras"]
    
    print(f"\n🎉 Procesamiento completado exitosamente")
    print(f"📊 Estadísticas generales:")
    print(f"   - Total de fragmentos: {len(todos_los_fragmentos)}")
    print(f"   - Total de palabras: {sum(f['palabras'] for f in todos_los_fragmentos)}")
    
    print(f"\n📚 Estadísticas por libro:")
    for libro, stats in libros_stats.items():
        print(f"   - {libro}: {stats['fragmentos']} fragmentos, {stats['palabras']} palabras")
    
    print(f"\n💾 Archivo generado:")
    print(f"   - Metadatos completos con texto en '{metadata_file}'")

if __name__ == "__main__":
    main()
