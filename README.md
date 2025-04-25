# Samara Discord Bot

Samara es un bot de Discord con memoria persistente y personalidad única. Utiliza tecnologías avanzadas como LangChain y Pinecone para mantener conversaciones contextuales y recordar interacciones pasadas.

## Características

- Memoria persistente de todas las interacciones
- Reconocimiento de usuarios por nickname
- Respuestas contextuales basadas en conversaciones anteriores
- Personalidad única y consistente
- Interacción natural en español

## Requisitos

- Node.js v16 o superior
- Una cuenta de Discord y token de bot
- Cuenta de OpenAI con API key
- Cuenta de Pinecone con API key

## Configuración

1. Clona el repositorio
2. Instala las dependencias:
```bash
npm install
```

3. Crea un archivo `.env` con las siguientes variables:
```
DISCORD_TOKEN=tu_token_de_discord
OPENAI_API_KEY=tu_token_de_openai
PINECONE_API_KEY=tu_api_key_de_pinecone
PINECONE_ENVIRONMENT=tu_environment_de_pinecone
PINECONE_INDEX=samara-memory
```

4. Inicia el bot:
```bash
node index.js
```

## Uso

Samara responderá cuando:
- Sea mencionada directamente (@Samara)
- Se escriba su nombre en el chat ("samara")
- Se hable sobre ella en el canal

## Estructura del Proyecto

- `index.js`: Archivo principal del bot
- `memory.js`: Sistema de memoria utilizando Pinecone
- `.env`: Configuración de variables de entorno

## Contribución

Si deseas contribuir al proyecto, por favor:
1. Haz fork del repositorio
2. Crea una rama para tu feature
3. Envía un pull request

## Licencia

MIT 