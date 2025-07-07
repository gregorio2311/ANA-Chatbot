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
        
        # Lista de secciones extraídas del índice (se actualizará dinámicamente)
        self.index_sections = []
        
        # Páginas que contienen el índice (se detectarán automáticamente)
        self.index_pages = set()

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando elementos repetitivos y innecesarios.
        
        Args:
            text (str): Texto original a limpiar
            
        Returns:
            str: Texto limpio
        """
        # Aplicar patrones de limpieza menos agresivos
        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalizar espacios pero preservar estructura
        text = re.sub(r' +', ' ', text)  # Múltiples espacios a uno
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Múltiples líneas vacías a dos
        
        # Remover líneas que son solo números de página
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # No incluir líneas que son solo números
            if not re.match(r'^\d+$', line):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
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
        Divide el texto en chunks con overlap mejorado.
        
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
        
        # Si el texto es muy corto, no dividir
        if len(text.split()) <= max_words:
            return [text]
        
        # Dividir por párrafos primero
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            words = paragraph.split()
            
            if len(words) <= max_words:
                # Párrafo completo cabe en un chunk
                if paragraph.strip():
                    chunks.append(paragraph.strip())
            else:
                # Dividir párrafo largo
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
                        if last_sentence_end > len(chunk_text) * 0.6:  # Si está en el último 40%
                            chunk_text = chunk_text[:last_sentence_end + 1]
                    
                    if chunk_text.strip():
                        chunks.append(chunk_text.strip())
                    
                    start += max_words - overlap_words
        
        # Si no se generaron chunks, usar el método original
        if not chunks:
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)
                
                if end < len(words):
                    last_sentence_end = max(
                        chunk_text.rfind('.'),
                        chunk_text.rfind(';'),
                        chunk_text.rfind(',')
                    )
                    if last_sentence_end > len(chunk_text) * 0.6:
                        chunk_text = chunk_text[:last_sentence_end + 1]
                
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                
                start += max_words - overlap_words
        
        return chunks

    def detect_index_pages(self, doc, source_name: str) -> set:
        """
        Detecta las páginas que contienen el índice del documento.
        
        Args:
            doc: Documento PyMuPDF
            source_name: Nombre del documento fuente
            
        Returns:
            set: Conjunto de números de página que contienen el índice
        """
        index_pages = set()
        
        # Para el manual específico, ignorar páginas 1-8 (índice)
        if "Complemento Anatomía Funcional Humana" in source_name:
            index_pages = set(range(8))  # Páginas 0-7 (índice)
            logger.info(f"📋 Configurado para ignorar páginas 1-8 del manual (índice)")
        
        return index_pages

    def extract_index_sections(self, doc) -> List[str]:
        """
        Extrae las secciones del índice para usarlas como palabras clave de referencia.
        
        Args:
            doc: Documento PyMuPDF
            
        Returns:
            List[str]: Lista de secciones extraídas del índice
        """
        sections = []
        
        for page_num in self.index_pages:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            if not text.strip():
                continue
            
            # Buscar líneas que parecen títulos de sección
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Patrones para detectar títulos de sección en el índice
                if (re.match(r'^\d+\.\s*[A-ZÁÉÍÓÚÜÑ]', line) or  # 1. Título
                    re.match(r'^[A-ZÁÉÍÓÚÜÑ][^.!?]{3,50}$', line) or  # Título sin numeración
                    re.match(r'^[IVX]+\.\s*[A-ZÁÉÍÓÚÜÑ]', line) or  # I. Título (romano)
                    re.match(r'^[A-ZÁÉÍÓÚÜÑ][^.!?]{3,30}\s+\d+$', line)):  # Título con número de página
                    
                    # Limpiar el título
                    clean_title = re.sub(r'^\d+\.\s*', '', line)  # Remover numeración
                    clean_title = re.sub(r'^[IVX]+\.\s*', '', clean_title)  # Remover numeración romana
                    clean_title = re.sub(r'\s+\d+$', '', clean_title)  # Remover número de página al final
                    clean_title = clean_title.strip()
                    
                    if clean_title and len(clean_title) > 3 and len(clean_title) < 100:
                        sections.append(clean_title)
        
        return list(set(sections))  # Remover duplicados

    def detect_titles_by_font_size(self, page) -> List[str]:
        """
        Detecta títulos basándose en el tamaño de fuente.
        
        Args:
            page: Página de PyMuPDF
            
        Returns:
            List[str]: Lista de títulos detectados
        """
        try:
            # Obtener bloques de texto con información de fuente
            blocks = page.get_text("dict")
            titles = []
            
            # Encontrar el tamaño de fuente más grande (probablemente títulos)
            font_sizes = []
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            if "size" in span:
                                font_sizes.append(span["size"])
            
            if not font_sizes:
                return []
            
            # El tamaño más grande es probablemente para títulos
            max_font_size = max(font_sizes)
            title_threshold = max_font_size * 0.8  # 80% del tamaño máximo (más permisivo)
            
            # Extraer texto con tamaño de fuente grande
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        is_title_line = False
                        max_span_size = 0
                        
                        for span in line.get("spans", []):
                            if "size" in span:
                                max_span_size = max(max_span_size, span["size"])
                            line_text += span.get("text", "")
                        
                        # Verificar si la línea tiene texto con tamaño grande
                        if max_span_size >= title_threshold and line_text.strip():
                            # Limpiar y validar el título
                            clean_title = line_text.strip()
                            # Verificar que sea un título válido
                            if (len(clean_title) < 150 and 
                                clean_title[0].isupper() and 
                                not clean_title.isdigit() and
                                len(clean_title.split()) < 15):  # No más de 15 palabras
                                titles.append(clean_title)
            
            return titles
            
        except Exception as e:
            logger.warning(f"Error detectando títulos por tamaño de fuente: {e}")
            return []

    def detect_sections_improved(self, page, text: str) -> List[Tuple[str, str]]:
        """
        Detecta secciones mejoradas usando información de fuente y patrones.
        
        Args:
            page: Página de PyMuPDF
            text (str): Texto de la página
            
        Returns:
            List[Tuple[str, str]]: Lista de (título, contenido)
        """
        # Detectar títulos por tamaño de fuente
        titles_by_font = self.detect_titles_by_font_size(page)
        
        # Patrones de palabras clave para títulos (expandidos)
        title_keywords = [
            "Anatomía", "Sistema", "Órgano", "Tejido", "Célula", "Músculo", 
            "Hueso", "Nervio", "Vaso", "Arteria", "Vena", "Ligamento", 
            "Tendón", "Cartílago", "Piel", "Cerebro", "Corazón", "Pulmón",
            "Hígado", "Riñón", "Estómago", "Intestino", "Columna", "Cráneo",
            "Tórax", "Abdomen", "Pelvis", "Extremidad", "Miembro", "Articulación",
            "Plano", "Región", "Cavidad", "Membrana", "Fascia", "Aponeurosis",
            "Concepto", "Clínico", "Fractura", "Luxación", "Lesión", "Síndrome",
            "Hernia", "Osteoporosis", "Várices", "Trombosis", "Inyección",
            "Plexo", "Nervio", "Vasos", "Hombro", "Codo", "Muñeca", "Rodilla",
            "Cadera", "Pierna", "Pie", "Dorso", "Axila", "Glútea", "Femoral"
        ]
        
        sections = []
        lines = text.split('\n')
        current_section = "General"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Verificar si la línea es un título
            is_title = False
            title_text = ""
            
            # 1. Verificar si está en la lista de títulos por fuente
            for title in titles_by_font:
                if (title.lower() in line.lower() or 
                    line.lower() in title.lower() or
                    any(word in line.lower() for word in title.lower().split())):
                    is_title = True
                    title_text = title
                    break
            
            # 2. Verificar patrones de palabras clave
            if not is_title:
                for keyword in title_keywords:
                    if keyword.lower() in line.lower() and len(line) < 120:
                        # Verificar que sea una línea de título
                        if (line[0].isupper() and 
                            not line.endswith('.') and 
                            len(line.split()) < 12 and
                            not any(char.isdigit() for char in line[:3])):  # No empiece con números
                            is_title = True
                            title_text = line
                            break
            
            # 3. Verificar patrones de numeración romana o arábiga
            if not is_title:
                # Patrones como "I.", "II.", "1.", "2.", etc.
                if re.match(r'^[IVX]+\.\s*[A-ZÁÉÍÓÚÜÑ]', line) or re.match(r'^\d+\.\s*[A-ZÁÉÍÓÚÜÑ]', line):
                    is_title = True
                    title_text = line
            
            # 4. Verificar líneas que parecen títulos por formato
            if not is_title:
                # Líneas cortas, en mayúsculas, sin punto final
                if (len(line) < 80 and 
                    line[0].isupper() and 
                    not line.endswith('.') and
                    len(line.split()) < 8 and
                    not any(char.isdigit() for char in line[:5])):
                    # Verificar que no sea solo una palabra común
                    common_words = ["el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de", "del", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "hasta", "desde", "durante", "según", "mediante", "contra", "ante", "bajo", "tras", "cabe", "so", "que", "cual", "quien", "cuyo", "donde", "cuando", "como", "cuanto", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas"]
                    if not any(word.lower() in common_words for word in line.split()):
                        is_title = True
                        title_text = line
            
            if is_title:
                # Guardar contenido anterior
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Actualizar sección actual
                current_section = title_text
                current_content = []
            else:
                current_content.append(line)
        
        # Agregar la última sección
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        # Si no se detectaron secciones, usar el texto completo como una sección
        if not sections and text.strip():
            sections.append(("Contenido General", text.strip()))
        
        return sections

    def extract_from_pdf(self, pdf_path: str, source_name: str) -> List[TextChunk]:
        """
        Extrae texto de un PDF usando enfoque simplificado: título principal + contenido completo.
        
        Args:
            pdf_path (str): Ruta al archivo PDF
            source_name (str): Nombre del documento fuente
            
        Returns:
            List[TextChunk]: Lista de chunks procesados
        """
        try:
            logger.info(f"📖 Procesando: {source_name}")
            doc = fitz.Document(pdf_path)
            
            # Detectar páginas del índice y extraer secciones de referencia
            self.index_pages = self.detect_index_pages(doc, source_name)
            if self.index_pages:
                logger.info(f"📋 Detectadas páginas de índice: {sorted(self.index_pages)}")
                self.index_sections = self.extract_index_sections(doc)
                logger.info(f"📝 Secciones extraídas del índice: {len(self.index_sections)}")
            
            all_chunks = []
            chunk_counter = 1
            total_chunks_per_document = 0  # Contador global para el documento
            
            # Primera pasada: contar total de chunks
            for page_num in range(len(doc)):
                # Saltar páginas del índice
                if page_num in self.index_pages:
                    logger.debug(f"⏭️ Saltando página de índice: {page_num + 1}")
                    continue
                
                page = doc.load_page(page_num)
                
                # Extraer texto por bloques con coordenadas para respetar el orden visual
                blocks = page.get_text("blocks")
                if not blocks:
                    continue
                
                # Ordenar bloques por posición vertical (y0) para respetar el orden visual
                blocks_sorted = sorted(blocks, key=lambda b: b[1])  # Ordenar por y0 (posición superior)
                
                # Concatenar bloques en el orden correcto
                text_ordered = "\n".join([b[4].strip() for b in blocks_sorted if b[4].strip()])
                if not text_ordered.strip():
                    continue
                
                # Limpiar texto
                clean_text = self.clean_text(str(text_ordered))
                if not clean_text.strip():
                    continue
                
                # Detectar título principal usando tamaño de fuente y secciones del índice
                titles_by_font = self.detect_titles_by_font_size(page)
                section_title = "General"
                
                if titles_by_font:
                    # Buscar coincidencias con secciones del índice
                    for title in titles_by_font:
                        for index_section in self.index_sections:
                            if (title.lower() in index_section.lower() or 
                                index_section.lower() in title.lower() or
                                any(word in title.lower() for word in index_section.lower().split())):
                                section_title = index_section
                                break
                        if section_title != "General":
                            break
                    # Si no hay coincidencias, usar el primer título detectado
                    if section_title == "General":
                        section_title = titles_by_font[0]
                
                # Todo el resto de texto se guarda como contenido (sin dividir en sub-secciones)
                section_content = clean_text
                
                # Crear chunks del contenido completo
                chunks = self.create_chunks(str(section_content))
                total_chunks_per_document += len(chunks)
            
            # Segunda pasada: crear chunks con información correcta
            for page_num in range(len(doc)):
                # Saltar páginas del índice
                if page_num in self.index_pages:
                    logger.debug(f"⏭️ Saltando página de índice: {page_num + 1}")
                    continue
                
                page = doc.load_page(page_num)
                
                # Extraer texto por bloques con coordenadas para respetar el orden visual
                blocks = page.get_text("blocks")
                if not blocks:
                    continue
                
                # Ordenar bloques por posición vertical (y0) para respetar el orden visual
                blocks_sorted = sorted(blocks, key=lambda b: b[1])  # Ordenar por y0 (posición superior)
                
                # Concatenar bloques en el orden correcto
                text_ordered = "\n".join([b[4].strip() for b in blocks_sorted if b[4].strip()])
                if not text_ordered.strip():
                    continue
                
                # Limpiar texto
                clean_text = self.clean_text(str(text_ordered))
                if not clean_text.strip():
                    continue
                
                # Detectar únicamente el título principal (usando el tamaño de fuente más grande)
                titles_by_font = self.detect_titles_by_font_size(page)
                if titles_by_font:
                    section_title = titles_by_font[0]  # Solo el más grande
                else:
                    section_title = "General"
                
                # Todo el resto de texto se guarda como contenido (sin dividir en sub-secciones)
                section_content = clean_text
                
                # Crear chunks del contenido completo
                chunks = self.create_chunks(str(section_content))
                
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
                    clean_section = clean_id(section_title)
                    chunk_id = f"{clean_source}_{clean_section}_{chunk_counter:03d}"
                    
                    # Crear objeto TextChunk
                    text_chunk = TextChunk(
                        id=chunk_id,
                        source=source_name,
                        section=section_title,
                        subsection=section_title,  # Usar el mismo título para subsección
                        page_number=page_num + 1,
                        text=chunk_text,
                        metadata={
                            "file_name": os.path.basename(pdf_path),
                            "chunk_index": chunk_idx + 1,
                            "total_chunks": len(chunks),  # Chunks en esta sección
                            "total_document_chunks": total_chunks_per_document,  # Total en documento
                            "word_count": len(chunk_text.split()),
                            "section": section_title,
                            "subsection": section_title
                        }
                    )
                    
                    all_chunks.append(text_chunk)
                    chunk_counter += 1
            
            doc.close()
            logger.info(f"✅ Extraídos {len(all_chunks)} chunks de {source_name}")
            logger.info(f"📊 Total de chunks en documento: {total_chunks_per_document}")
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