# ANA-Chatbot
Chatbot de anatomía con búsqueda semántica

## 🔐 Configuración de Credenciales

### 1. Variables de Entorno
Copia el archivo `.env.example` como `.env` y configura tus credenciales:

```bash
cp .env.example .env
```

Edita el archivo `.env` con tus credenciales de Pinecone:
```
PINECONE_API_KEY=tu_api_key_aqui
PINECONE_HOST=tu_host_aqui
PINECONE_INDEX_NAME=tu_index_name_aqui
```

### 2. Instalación
```bash
# Activar entorno virtual
.\venv311\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Uso
```bash
# Ejecutar chatbot
python scripts/consulta.py
```

## ⚠️ Seguridad
- **NUNCA** subas el archivo `.env` a Git
- Las credenciales están protegidas en `.gitignore`
- Usa variables de entorno para credenciales en producción

## 📁 Estructura del Proyecto
- `scripts/` - Scripts principales del chatbot
- `data/` - Documentos fuente (no incluidos en Git)
- `fragmentos/` - Fragmentos procesados (no incluidos en Git)
- `embeddings.pkl` - Embeddings generados (no incluido en Git)
