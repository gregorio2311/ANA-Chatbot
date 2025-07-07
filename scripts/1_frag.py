"""
EXTRACTOR MEJORADO DE TEXTO - ANA-CHATBOT
=========================================

Este script implementa una extracción de texto mejorada usando PyMuPDF (fitz)
con las siguientes mejoras:

1. Extracción por secciones usando el índice
2. División en subsecciones usando títulos
3. Limpieza inteligente de texto
4. Chunking optimizado para BGE-m3
5. Formato JSON estructurado

FUNCIONALIDADES:
- Extrae texto de PDFs usando PyMuPDF
- Detecta secciones y subsecciones automáticamente
- Limpia texto eliminando elementos repetitivos
- Crea chunks de 500-1000 tokens con overlap del 20%
- Genera metadatos estructurados
- Soporta tanto diapositivas como manuales

REQUISITOS:
- PyMuPDF (fitz) instalado
- PDFs en data/ana_fun/
- Estructura: diapos/ (diapositivas) y man/ (manuales)

USO:
    python scripts/1_frag_mejorado.py

FLUJO DE TRABAJO:
    1. Extraer texto por secciones → 1_frag_mejorado.py (este script)
    2. Generar embeddings → 2_embeddings.py
    3. Subir a Pinecone → 3_pinecone_u.py
    4. Usar chatbot → 4_consulta.py

AUTOR: Equipo de desarrollo ANA-Chatbot
FECHA: 2024
"""

import os
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Estructura para representar un chunk de texto procesado."""
    id: str
    source: str
    section: str
    subsection: str
    page_number: int
    text: str
    metadata: Dict[str, Any]

class TextExtractor:
    """Extractor de texto mejorado usando PyMuPDF."""
    
    def __init__(self):
        self.patterns_to_remove = [
            # Encabezados y pies de página repetitivos
            r'Netter.*Atlas of Human Anatomy',
            r'Complemento Anatomía Funcional Humana',
            r'Universidad.*Anáhuac',
            r'Facultad.*Medicina',
            
            # URLs
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            
            # Créditos de figuras repetitivos
            r'Netter, F\. H\.: Atlas of Human Anatomy.*?\.',
            r'Figura \d+.*?Netter',
            
            # Numeración de páginas innecesaria
            r'Página \d+',
            r'Page \d+',
            
            # Caracteres especiales problemáticos
            r'[^\w\s\.\,\;\:\!\?\-\(\)áéíóúüñÁÉÍÓÚÜÑ]',
        ]
        
        # Patrones para detectar títulos de secciones
        self.section_patterns = [
            r'^(\d+\.\s*)([A-ZÁÉÍÓÚÜÑ][^.!?]*?)(?=\n|$)',
            r'^([A-ZÁÉÍÓÚÜÑ][^.!?]{3,50})(?=\n|$)',
            r'^(\d+\.\d+\s*)([A-ZÁÉÍÓÚÜÑ][^.!?]*?)(?=\n|$)',
        ]
        
        # Patrones para detectar subsecciones
        self.subsection_patterns = [
            r'^([a-záéíóúüñ][^.!?]{3,30})(?=\n|$)',
            r'^(\d+\.\d+\.\d+\s*)([A-ZÁÉÍÓÚÜÑ][^.!?]*?)(?=\n|$)',
        ]

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando elementos repetitivos y innecesarios.
        
        Args:
            text (str): Texto original a limpiar
            
        Returns:
            str: Texto limpio
        """
        # Aplicar patrones de limpieza
        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def detect_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Detecta secciones y subsecciones en el texto.
        
        Args:
            text (str): Texto completo
            
        Returns:
            List[Tuple[str, str, int]]: Lista de (título, contenido, página)
        """
        sections = []
        lines = text.split('\n')
        current_section = "General"
        current_subsection = "Introducción"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detectar sección principal
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    if current_content:
                        sections.append((current_section, current_subsection, '\n'.join(current_content)))
                    current_section = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    current_subsection = "General"
                    current_content = []
                    break
            else:
                # Detectar subsección
                for pattern in self.subsection_patterns:
                    match = re.match(pattern, line)
                    if match:
                        if current_content:
                            sections.append((current_section, current_subsection, '\n'.join(current_content)))
                        current_subsection = match.group(2) if len(match.groups()) > 1 else match.group(1)
                        current_content = []
                        break
                else:
                    current_content.append(line)
        
        # Agregar la última sección
        if current_content:
            sections.append((current_section, current_subsection, '\n'.join(current_content)))
        
        return sections

    def create_chunks(self, text: str, max_tokens: int = 800, overlap: float = 0.2) -> List[str]:
        """
        Divide el texto en chunks con overlap.
        
        Args:
            text (str): Texto a dividir
            max_tokens (int): Máximo número de tokens por chunk
            overlap (float): Porcentaje de overlap entre chunks
            
        Returns:
            List[str]: Lista de chunks
        """
        # Estimación simple: 1 token ≈ 1.3 palabras
        max_words = int(max_tokens / 1.3)
        overlap_words = int(max_words * overlap)
        
        words = text.split()
        chunks = []
        
        if len(words) <= max_words:
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            
            # Intentar cortar en una oración completa
            chunk_text = ' '.join(chunk_words)
            if end < len(words):
                # Buscar el último punto, coma o punto y coma
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind(';'),
                    chunk_text.rfind(',')
                )
                if last_sentence_end > len(chunk_text) * 0.7:  # Si está en el último 30%
                    chunk_text = chunk_text[:last_sentence_end + 1]
            
            chunks.append(chunk_text.strip())
            start += max_words - overlap_words
        
        return chunks

    def extract_from_pdf(self, pdf_path: str, source_name: str) -> List[TextChunk]:
        """
        Extrae texto de un PDF y lo procesa en chunks estructurados.
        
        Args:
            pdf_path (str): Ruta al archivo PDF
            source_name (str): Nombre del documento fuente
            
        Returns:
            List[TextChunk]: Lista de chunks procesados
        """
        try:
            logger.info(f"📖 Procesando: {source_name}")
            doc = fitz.Document(pdf_path)
            
            all_chunks = []
            chunk_counter = 1
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Limpiar texto
                clean_text = self.clean_text(str(text))
                
                # Detectar secciones
                sections = self.detect_sections(clean_text)
                
                for section, subsection, content in sections:
                    if not str(content).strip():
                        continue
                    
                    # Crear chunks del contenido
                    chunks = self.create_chunks(str(content))
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        if not chunk_text.strip():
                            continue
                        
                        # Crear ID único compatible con Pinecone (solo ASCII)
                        def clean_id(text):
                            """Limpia texto para crear ID compatible con Pinecone"""
                            import unicodedata
                            # Normalizar caracteres Unicode
                            text = unicodedata.normalize('NFKD', text)
                            # Remover acentos y caracteres especiales
                            text = ''.join(c for c in text if c.isascii() and c.isalnum() or c in '_-')
                            # Reemplazar espacios y caracteres problemáticos
                            text = text.replace(' ', '_').replace('-', '_').replace(',', '').replace('.', '')
                            # Limitar longitud y asegurar que empiece con letra
                            if text and not text[0].isalpha():
                                text = 'f_' + text
                            return text[:50]  # Limitar longitud
                        
                        clean_source = clean_id(source_name)
                        clean_section = clean_id(section)
                        chunk_id = f"{clean_source}_{clean_section}_{chunk_counter:03d}"
                        
                        # Crear objeto TextChunk
                        text_chunk = TextChunk(
                            id=chunk_id,
                            source=source_name,
                            section=section,
                            subsection=subsection,
                            page_number=page_num + 1,
                            text=chunk_text,
                            metadata={
                                "file_name": os.path.basename(pdf_path),
                                "chunk_index": chunk_idx + 1,
                                "total_chunks": len(chunks),
                                "word_count": len(chunk_text.split()),
                                "section": section,
                                "subsection": subsection
                            }
                        )
                        
                        all_chunks.append(text_chunk)
                        chunk_counter += 1
            
            doc.close()
            logger.info(f"✅ Extraídos {len(all_chunks)} chunks de {source_name}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando {source_name}: {e}")
            return []

    def process_ana_fun_content(self) -> List[Dict[str, Any]]:
        """
        Procesa todo el contenido de la carpeta ana_fun.
        
        Returns:
            List[Dict[str, Any]]: Lista de chunks en formato JSON
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        ana_fun_dir = os.path.join(project_root, "data", "ana_fun")
        
        if not os.path.exists(ana_fun_dir):
            logger.error(f"❌ No se encontró la carpeta: {ana_fun_dir}")
            return []
        
        all_chunks = []
        
        # Procesar diapositivas
        diapos_dir = os.path.join(ana_fun_dir, "diapos")
        if os.path.exists(diapos_dir):
            logger.info("📚 Procesando diapositivas...")
            for file in os.listdir(diapos_dir):
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(diapos_dir, file)
                    source_name = f"Diapositivas - {os.path.splitext(file)[0]}"
                    chunks = self.extract_from_pdf(pdf_path, source_name)
                    all_chunks.extend(chunks)
        
        # Procesar manual
        man_dir = os.path.join(ana_fun_dir, "man")
        if os.path.exists(man_dir):
            logger.info("📖 Procesando manual...")
            for file in os.listdir(man_dir):
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(man_dir, file)
                    source_name = "Complemento Anatomía Funcional Humana"
                    chunks = self.extract_from_pdf(pdf_path, source_name)
                    all_chunks.extend(chunks)
        
        # Convertir a formato JSON
        json_chunks = []
        for chunk in all_chunks:
            json_chunk = {
                "id": chunk.id,
                "source": chunk.source,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            json_chunks.append(json_chunk)
        
        return json_chunks

def main():
    """
    Función principal que ejecuta el procesamiento completo.
    """
    try:
        logger.info("🚀 Iniciando extracción mejorada de texto...")
        
        extractor = TextExtractor()
        chunks = extractor.process_ana_fun_content()
        
        if not chunks:
            logger.error("❌ No se pudieron procesar chunks")
            return False
        
        # Guardar en archivo JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_file = os.path.join(project_root, "data", "fragmentos_mejorados.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Estadísticas
        sources = set(chunk["source"] for chunk in chunks)
        sections = set(chunk["section"] for chunk in chunks)
        total_words = sum(chunk["metadata"]["word_count"] for chunk in chunks)
        
        logger.info(f"🎉 Procesamiento completado exitosamente")
        logger.info(f"📊 Estadísticas:")
        logger.info(f"   - Total de chunks: {len(chunks)}")
        logger.info(f"   - Total de palabras: {total_words}")
        logger.info(f"   - Fuentes: {len(sources)}")
        logger.info(f"   - Secciones: {len(sections)}")
        logger.info(f"💾 Archivo guardado: {output_file}")
        
        # Mostrar algunas estadísticas por fuente
        for source in sources:
            source_chunks = [c for c in chunks if c["source"] == source]
            source_words = sum(c["metadata"]["word_count"] for c in source_chunks)
            logger.info(f"   - {source}: {len(source_chunks)} chunks, {source_words} palabras")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 