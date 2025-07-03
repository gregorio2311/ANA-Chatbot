"""
LIMPIEZA COMPLETA DE PINECONE - ANA-CHATBOT
==========================================

Este script elimina completamente todos los vectores de la base de datos Pinecone.
Útil para limpiar datos antiguos o corrompidos antes de una nueva subida.

FUNCIONALIDADES:
- Conecta con Pinecone usando credenciales seguras
- Elimina todos los vectores del índice especificado
- Proporciona confirmación antes de la eliminación
- Muestra estadísticas antes y después de la limpieza

REQUISITOS:
- Variables de entorno configuradas (.env):
  * PINECONE_API_KEY: Clave API de Pinecone
  * PINECONE_HOST: URL del host de Pinecone
  * PINECONE_INDEX_NAME: Nombre del índice (opcional, default: "ana")

USO:
    python scripts/limpiar_pinecone.py

EJEMPLO DE SALIDA:
    🚀 Iniciando limpieza completa de Pinecone...
    📊 Estadísticas actuales del índice 'ana':
       - Total de vectores: 3524
       - Dimensiones: 1024
    ⚠️ ADVERTENCIA: Esto eliminará TODOS los vectores del índice
    ¿Está seguro de continuar? (sí/no): sí
    🗑️ Eliminando todos los vectores...
    ✅ Limpieza completada exitosamente
    📊 Estadísticas después de la limpieza:
       - Total de vectores: 0

DEPENDENCIAS:
- pinecone: Para conexión con base de datos vectorial
- python-dotenv: Para cargar variables de entorno

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

import os
import sys
from pinecone import Pinecone
from dotenv import load_dotenv

def conectar_pinecone(api_key: str, index_name: str):
    """
    Conecta a Pinecone serverless usando el SDK v3.
    """
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Verificar que el índice existe
        stats = index.describe_index_stats()
        print(f"✅ Conectado al índice '{index_name}'")
        print(f"📊 Estadísticas actuales:")
        print(f"   - Total de vectores: {stats.get('total_vector_count', 0)}")
        print(f"   - Dimensiones: {stats.get('dimension', 'N/A')}")
        
        return index

    except Exception as e:
        print(f"❌ Error conectando a Pinecone: {e}")
        return None

def mostrar_estadisticas(index):
    """
    Muestra estadísticas detalladas del índice.
    """
    try:
        stats = index.describe_index_stats()
        print(f"\n📊 Estadísticas del índice:")
        print(f"   - Total de vectores: {stats.get('total_vector_count', 0)}")
        print(f"   - Dimensiones: {stats.get('dimension', 'N/A')}")
        print(f"   - Métrica: {stats.get('metric', 'N/A')}")
        print(f"   - Tipo de vector: {stats.get('vector_type', 'N/A')}")
        
        # Mostrar información por namespace
        namespaces = stats.get('namespaces', {})
        if namespaces:
            print(f"   - Namespaces:")
            for ns, ns_stats in namespaces.items():
                print(f"     * '{ns}': {ns_stats.get('vector_count', 0)} vectores")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return None

def limpiar_indice(index):
    """
    Elimina todos los vectores del índice.
    """
    try:
        print("🗑️ Eliminando todos los vectores...")
        
        # Eliminar todos los vectores usando delete_all()
        index.delete(delete_all=True)
        
        print("✅ Limpieza completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error durante la limpieza: {e}")
        return False

def confirmar_limpieza():
    """
    Solicita confirmación del usuario antes de proceder.
    """
    print("\n⚠️ ADVERTENCIA: Esto eliminará TODOS los vectores del índice")
    print("💡 Esta acción no se puede deshacer")
    
    while True:
        respuesta = input("¿Está seguro de continuar? (sí/no): ").lower().strip()
        
        if respuesta in ['sí', 'si', 's', 'yes', 'y']:
            return True
        elif respuesta in ['no', 'n']:
            return False
        else:
            print("Por favor, responda 'sí' o 'no'")

def main():
    """
    Función principal que ejecuta la limpieza completa de Pinecone.
    """
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Configuración
        API_KEY = os.getenv("PINECONE_API_KEY")
        HOST = os.getenv("PINECONE_HOST")
        INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ana")
        
        # Validar credenciales
        if not API_KEY or not HOST:
            print("❌ Error: Faltan variables de entorno")
            print("💡 Configura PINECONE_API_KEY y PINECONE_HOST en tu archivo .env")
            return False
        
        print("🚀 Iniciando limpieza completa de Pinecone...")
        
        # Conectar a Pinecone
        index = conectar_pinecone(API_KEY, INDEX_NAME)
        if index is None:
            return False
        
        # Mostrar estadísticas actuales
        stats_antes = mostrar_estadisticas(index)
        if not stats_antes:
            return False
        
        # Confirmar con el usuario
        if not confirmar_limpieza():
            print("❌ Operación cancelada por el usuario")
            return False
        
        # Realizar limpieza
        if limpiar_indice(index):
            # Mostrar estadísticas después
            print("\n📊 Estadísticas después de la limpieza:")
            stats_despues = mostrar_estadisticas(index)
            
            if stats_despues:
                vectores_eliminados = stats_antes.get('total_vector_count', 0) - stats_despues.get('total_vector_count', 0)
                print(f"\n🎉 Resumen:")
                print(f"   - Vectores eliminados: {vectores_eliminados}")
                print(f"   - Índice limpiado exitosamente")
            
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Proceso completado exitosamente")
    else:
        print("\n❌ El proceso falló")
        sys.exit(1) 