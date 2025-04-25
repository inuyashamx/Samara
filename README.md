# Samara Discord IA

<p align="center">
  <img src="https://img.itch.zone/aW1nLzIwNjY0MDMwLnBuZw==/original/MY5uVE.png" alt="Samara Bot Logo" width="200"/>
</p>

Samara es un avanzado bot de Discord que actúa como una persona real, con memoria persistente y personalidad única. Diseñado para interactuar de manera natural en servidores de Discord, Samara se comporta como una mujer joven de 20 años, manteniendo conversaciones fluidas y recordando detalles personales sobre los usuarios. Utiliza tecnologías de vanguardia como LangChain, OpenAI y Pinecone para mantener conversaciones contextuales y recordar interacciones pasadas con los usuarios.

## Índice

- [Características](#características)
- [Arquitectura Técnica](#arquitectura-técnica)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Sistema de Memoria](#sistema-de-memoria)
- [Personalización](#personalización)
- [Solución de Problemas](#solución-de-problemas)
- [Contribución](#contribución)
- [Licencia](#licencia)

## Características

- **Comportamiento Humano**: Interactúa como una persona real de 20 años, no como un bot tradicional
- **Personalidad Consistente**: Mantiene una personalidad de mujer joven, seria pero conversacional
- **Interacción Natural**: Responde en español con un estilo conversacional fluido, haciendo preguntas de seguimiento
- **Memoria Persistente**: Recuerda conversaciones y detalles específicos sobre los usuarios
- **Extracción de Hechos**: Identifica y almacena información relevante sobre los usuarios
- **Búsqueda Semántica**: Encuentra mensajes relevantes basados en el contexto de la conversación
- **Monitoreo de Canales**: Registra mensajes de canales específicos para construir contexto
- **Integración con Pinecone**: Almacenamiento vectorial para búsqueda semántica eficiente
- **Respuestas Contextuales**: Genera respuestas basadas en el historial de conversaciones
- **Detección de Menciones**: Responde cuando es mencionada o cuando se habla sobre ella

## Arquitectura Técnica

Samara está construida sobre una arquitectura moderna que integra varias tecnologías:

- **Discord.js**: Interacción con la API de Discord
- **OpenAI**: Generación de respuestas naturales y extracción de información
- **LangChain**: Framework para aplicaciones basadas en LLMs
- **Pinecone**: Base de datos vectorial para almacenamiento y búsqueda semántica
- **Node.js**: Entorno de ejecución para JavaScript

### Diagrama de Flujo

```
Usuario → Discord → Samara Bot → Procesamiento de Mensaje
                                   ↓
                 ┌────────────────┴─────────────────┐
                 ↓                                   ↓
          Extracción de Hechos             Búsqueda de Contexto
                 ↓                                   ↓
            LangMem (JSON)                     Pinecone (Vectores)
                 ↓                                   ↓
                 └────────────────┬─────────────────┘
                                   ↓
                          Generación de Respuesta
                                   ↓
                               OpenAI
                                   ↓
                          Respuesta al Usuario
```

## Requisitos

- **Node.js**: v16.0.0 o superior
- **NPM**: v7.0.0 o superior
- **Discord Bot**: Token de aplicación de Discord
- **OpenAI API**: Cuenta con acceso a la API y clave API
- **Pinecone**: Cuenta con índice configurado (dimensión 1024)
- **Almacenamiento**: Mínimo 1GB para dependencias y datos

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/samara.git
   cd samara
   ```

2. Instala las dependencias:
   ```bash
   npm install
   ```

3. Crea la estructura de directorios necesaria:
   ```bash
   mkdir -p data
   ```

## Configuración

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
# Discord
DISCORD_TOKEN=tu_token_de_discord

# OpenAI
OPENAI_API_KEY=tu_token_de_openai

# Pinecone
PINECONE_API_KEY=tu_api_key_de_pinecone
PINECONE_ENVIRONMENT=tu_environment_de_pinecone
PINECONE_INDEX=xpellit

# MongoDB (opcional)
MONGODB_URI=tu_uri_de_mongodb
```

### Configuración de Canales

Por defecto, Samara monitorea los canales `chat-general` e `impostor`. Para cambiar esto, modifica las siguientes líneas en `index.js`:

```javascript
// Canales a monitorear
const MONITORED_CHANNELS = ['chat-general', 'impostor'];
```

### Configuración de Personalidad

La personalidad de Samara puede ajustarse modificando el prompt del sistema en `index.js`. Busca la sección que comienza con:

```javascript
// Construir el prompt del sistema
const systemPrompt = `Eres Samara, una mujer de 20 años...
```

## Uso

### Iniciar el Bot

```bash
node index.js
```

Para mantener el bot ejecutándose en segundo plano, puedes usar herramientas como PM2:

```bash
npm install -g pm2
pm2 start index.js --name samara
```

### Interacción con Samara

Samara responderá cuando:

- Sea mencionada directamente: `@Samara hola`
- Se le haga una pregunta directa sobre información que conoce
- Se le pregunte sobre conversaciones pasadas en los canales monitoreados

### Comandos Principales

Samara no utiliza comandos con prefijos, sino que responde a lenguaje natural:

- **Preguntas sobre usuarios**: `@Samara qué sabes de [usuario]?`
- **Preguntas sobre conversaciones**: `@Samara qué se habló ayer en el chat?`
- **Compartir información**: `@Samara soy programador y me gusta el anime`

## Estructura del Proyecto

```
samara/
├── index.js           # Punto de entrada principal
├── langmem.js         # Sistema de memoria local
├── memory.js          # Integración con Pinecone
├── .env               # Variables de entorno
├── package.json       # Dependencias
├── data/              # Datos persistentes
│   ├── memories.json  # Memorias de usuarios
│   ├── facts.json     # Hechos sobre usuarios
│   └── relationships.json # Relaciones entre usuarios
└── README.md          # Documentación
```

## Sistema de Memoria

Samara utiliza un sistema de memoria dual:

### 1. Memoria Vectorial (Pinecone)

Almacena los mensajes completos como vectores de embeddings, permitiendo búsquedas semánticas para encontrar mensajes similares o relevantes a una consulta actual.

### 2. Memoria Estructurada (LangMem)

Almacena hechos específicos sobre usuarios en formato JSON. Estos hechos son extraídos automáticamente de las conversaciones y organizados para facilitar su recuperación.

### Tipos de Datos Almacenados

- **Mensajes**: Texto completo, autor, canal, timestamp
- **Hechos**: Información específica sobre usuarios (profesión, gustos, etc.)
- **Relaciones**: Conexiones entre usuarios (opcional)

## Personalización

### Ajuste de Personalidad

Modifica el prompt del sistema en `index.js` para cambiar la personalidad de Samara. Por defecto, Samara está configurada para comportarse como una mujer joven de 20 años con las siguientes características:

- **Seria pero conversacional**: Mantiene un tono serio pero sabe cómo mantener una conversación fluida
- **Preguntas de seguimiento**: Hace preguntas para mantener la conversación activa
- **Respuestas breves**: No escribe respuestas excesivamente largas
- **Comportamiento humano**: No habla como una IA ni usa frases como "como entidad digital" o "mi función es"
- **Miembro del grupo**: Actúa como un miembro más del grupo, no como un asistente o bot

Para modificar estos aspectos, busca la sección del prompt del sistema que comienza con:

```javascript
// Construir el prompt del sistema
const systemPrompt = `Eres Samara, una mujer de 20 años...
```

### Ejemplo de Interacción Natural

Samara está diseñada para mantener conversaciones que parezcan naturales. Por ejemplo:

**Usuario**: "@Samara ¿qué opinas sobre los videojuegos?"

**Samara**: "Me gustan bastante, sobre todo los de aventura y RPG. He estado jugando Elden Ring últimamente. ¿Tú juegas algo?"

En lugar de:

**Bot tradicional**: "Como asistente, puedo informarte que los videojuegos son una forma popular de entretenimiento digital. ¿Puedo ayudarte con algo más sobre este tema?"

### Canales Monitoreados

Cambia la lista `MONITORED_CHANNELS` en `index.js` para monitorear diferentes canales.

### Extracción de Hechos

Ajusta la función `extractFacts` en `index.js` para modificar cómo se extraen los hechos de los mensajes.

## 🔧 Solución de Problemas

### Problemas Comunes

1. **Error de conexión a Discord**:
   - Verifica que el token de Discord sea válido
   - Asegúrate de que el bot tenga los permisos necesarios

2. **Error de OpenAI**:
   - Verifica que la API key sea válida
   - Comprueba que tengas saldo suficiente en tu cuenta

3. **Error de Pinecone**:
   - Verifica las credenciales de Pinecone
   - Asegúrate de que el índice exista y tenga la dimensión correcta (1024)

4. **Memoria no funciona**:
   - Verifica que el directorio `data/` exista y tenga permisos de escritura
   - Comprueba los logs para errores específicos

### Logs

Los logs se muestran en la consola y pueden ayudar a diagnosticar problemas. Para guardar logs en un archivo:

```bash
node index.js > samara.log 2>&1
```

## Contribución

¡Las contribuciones son bienvenidas! Si deseas mejorar Samara:

1. Haz fork del repositorio
2. Crea una rama para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. Realiza tus cambios y haz commit: `git commit -m 'Añadir nueva funcionalidad'`
4. Sube tus cambios: `git push origin feature/nueva-funcionalidad`
5. Envía un pull request

### Guía de Estilo

- Usa ESLint para mantener un estilo de código consistente
- Documenta las funciones nuevas
- Añade pruebas para nuevas funcionalidades cuando sea posible

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

Desarrollado con ❤️ para la comunidad de Discord.
