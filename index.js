const { Client, GatewayIntentBits, Events } = require('discord.js');
const { ChatOpenAI } = require('@langchain/openai');
const { Pinecone } = require('@pinecone-database/pinecone');
const { PineconeStore } = require('@langchain/pinecone');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { BufferMemory } = require("langchain/memory");
const LangMem = require('./langmem');
const path = require('path'); // Import the path module
const fs = require('fs'); // Import the fs module
require('dotenv').config();

// Configuración del cliente de Discord
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.GuildMembers,
  ],
});

// Configuración de OpenAI
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4",
  temperature: 0.7
});

// Configuración de embeddings con adaptador para reducir dimensiones
class DimensionReducedEmbeddings extends OpenAIEmbeddings {
  constructor(config) {
    super(config);
    this.targetDimension = config.targetDimension || 1024;
    console.log(`Inicializando embeddings con reducción de dimensiones a ${this.targetDimension}`);
  }

  // Sobrescribir el método de embedQuery para reducir dimensiones
  async embedQuery(text) {
    const fullEmbedding = await super.embedQuery(text);
    return this.reduceDimensions(fullEmbedding);
  }

  // Sobrescribir el método de embedDocuments para reducir dimensiones
  async embedDocuments(documents) {
    const fullEmbeddings = await super.embedDocuments(documents);
    return fullEmbeddings.map(embedding => this.reduceDimensions(embedding));
  }

  // Método para reducir dimensiones
  reduceDimensions(embedding) {
    if (embedding.length <= this.targetDimension) {
      console.log(`No se necesita reducción: ${embedding.length} <= ${this.targetDimension}`);
      return embedding;
    }
    
    console.log(`Reduciendo dimensiones de ${embedding.length} a ${this.targetDimension}`);
    
    // Método simple: tomar los primeros N elementos
    // Hay métodos más sofisticados como PCA, pero esto es más rápido
    const reduced = embedding.slice(0, this.targetDimension);
    return reduced;
  }
}

// Instanciar los embeddings con reducción de dimensiones
const embeddings = new DimensionReducedEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-embedding-ada-002",
  targetDimension: 1024
});

// Configuración de memorias
let vectorStore;
const conversationMemory = new BufferMemory({ 
  returnMessages: true, 
  memoryKey: "chat_history",
  inputKey: "input"
});
const langMem = new LangMem({
  dataDir: path.join(__dirname, 'data') // Update the LangMem initialization to use the data directory
});

// Memoria local de mensajes recientes
let recentMessages = [];

// Caché global de conversaciones recientes entre usuarios
let recentInteractions = new Map();

// Función para guardar un mensaje en la memoria local
function saveMessageToLocalMemory(message) {
  if (!message || !message.content) return;
  
  // Crear objeto de mensaje para la memoria local
  const messageObj = {
    content: message.content,
    author: message.author.username,
    authorId: message.author.id,
    authorTag: message.author.tag,
    channel: message.channel.name,
    timestamp: new Date().toISOString(),
    mentions: []
  };
  
  // Añadir menciones si existen
  if (message.mentions && message.mentions.users) {
    message.mentions.users.forEach(user => {
      if (user.id !== client.user.id) { // No incluir a Samara
        messageObj.mentions.push({
          id: user.id,
          username: user.username,
          tag: user.tag
        });
      }
    });
  }
  
  // Añadir mensaje a la memoria reciente (al inicio del array)
  recentMessages.unshift(messageObj);
  
  // Mantener solo los últimos 200 mensajes
  if (recentMessages.length > 200) {
    recentMessages = recentMessages.slice(0, 200);
  }
  
  // Guardar la memoria local en un archivo para persistencia
  try {
    fs.writeFileSync(
      path.join(__dirname, 'data', 'recent_messages.json'), 
      JSON.stringify(recentMessages, null, 2)
    );
  } catch (error) {
    console.error('Error al guardar mensajes recientes:', error);
  }
}

// Función para registrar interacción entre usuarios
function trackUserInteraction(userId, username, interactionType, targetUserId, targetUsername, content) {
  // Crear clave para la interacción bidireccional
  const interactionKey1 = `${userId}-${targetUserId}`;
  const interactionKey2 = `${targetUserId}-${userId}`;
  
  // Crear objeto de interacción
  const interaction = {
    userId,
    username,
    targetUserId,
    targetUsername,
    type: interactionType, // 'message', 'mention', 'reply', etc.
    content,
    timestamp: new Date().toISOString()
  };
  
  // Guardar en ambas direcciones para facilitar búsquedas
  recentInteractions.set(interactionKey1, interaction);
  
  // Crear versión inversa de la interacción
  const reverseInteraction = {
    userId: targetUserId,
    username: targetUsername,
    targetUserId: userId,
    targetUsername: username,
    type: 'received_' + interactionType,
    content,
    timestamp: new Date().toISOString()
  };
  
  recentInteractions.set(interactionKey2, reverseInteraction);
  
  // Guardar en archivo para persistencia
  try {
    const interactionsObj = {};
    for (const [key, value] of recentInteractions.entries()) {
      interactionsObj[key] = value;
    }
    
    fs.writeFileSync(
      path.join(__dirname, 'data', 'recent_interactions.json'),
      JSON.stringify(interactionsObj, null, 2)
    );
    
    console.log(`Interacción registrada entre ${username} y ${targetUsername}`);
  } catch (error) {
    console.error('Error al guardar interacciones recientes:', error);
  }
}

// Función para cargar mensajes recientes al inicio
function loadRecentMessages() {
  try {
    const filePath = path.join(__dirname, 'data', 'recent_messages.json');
    if (fs.existsSync(filePath)) {
      const data = fs.readFileSync(filePath, 'utf8');
      recentMessages = JSON.parse(data);
      console.log(`Cargados ${recentMessages.length} mensajes recientes desde archivo`);
    }
  } catch (error) {
    console.error('Error al cargar mensajes recientes:', error);
    recentMessages = [];
  }
}

// Función para cargar interacciones recientes al inicio
function loadRecentInteractions() {
  try {
    const filePath = path.join(__dirname, 'data', 'recent_interactions.json');
    if (fs.existsSync(filePath)) {
      const data = fs.readFileSync(filePath, 'utf8');
      const interactionsObj = JSON.parse(data);
      for (const [key, value] of Object.entries(interactionsObj)) {
        recentInteractions.set(key, value);
      }
      console.log(`Cargadas ${recentInteractions.size} interacciones recientes desde archivo`);
    }
  } catch (error) {
    console.error('Error al cargar interacciones recientes:', error);
    recentInteractions = new Map();
  }
}

// Función para identificar a un usuario
function getUserIdentifier(user) {
  return {
    id: user.id,
    tag: user.tag,
    username: user.username,
    discriminator: user.discriminator
  };
}

// Inicialización de Pinecone y vectorStore
async function initialize() {
  try {
    console.log('Inicializando Pinecone...');
    
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    // Asegurarse de que el nombre del índice esté configurado
    const indexName = process.env.PINECONE_INDEX || 'xpellit';
    console.log(`Conectando al índice: ${indexName}`);

    const index = pinecone.Index(indexName);
    
    // Verificar estadísticas del índice
    try {
      const stats = await index.describeIndexStats();
      console.log('Estadísticas del índice Pinecone:');
      console.log(JSON.stringify(stats, null, 2));
      console.log(`- Vectores totales: ${stats.totalRecordCount || 'No disponible'}`);
      console.log(`- Dimensión: ${stats.dimension || 'No disponible'}`);
      
      // Mostrar namespaces y conteos si están disponibles
      if (stats.namespaces) {
        console.log('- Namespaces:');
        for (const [namespace, data] of Object.entries(stats.namespaces)) {
          console.log(`  * ${namespace}: ${data.recordCount || 0} vectores`);
        }
      } else {
        console.log('- Namespaces: ninguno o no disponible');
      }
    } catch (statsError) {
      console.error('Error al obtener estadísticas del índice:', statsError);
    }

    // Crear el vectorStore
    console.log('Creando VectorStore...');
    vectorStore = await PineconeStore.fromExistingIndex(
      embeddings,
      { 
        pineconeIndex: index,
        textKey: 'text',
        namespace: 'samara-memory' // Usar específicamente el namespace samara-memory
      }
    );

    // Insertar un documento de prueba para verificar que funciona
    await vectorStore.addDocuments([{
      pageContent: "Documento de inicialización de Samara",
      metadata: {
        source: "system",
        author: "System",
        type: "initialization",
        timestamp: new Date().toISOString()
      }
    }]);

    console.log('Inicialización completada con éxito');
    
    // Programar una tarea para mostrar estadísticas periódicamente
    setInterval(async () => {
      try {
        const stats = await index.describeIndexStats();
        console.log(`[${new Date().toISOString()}] Estadísticas actuales del índice:`);
        
        // Mostrar conteo total de vectores
        console.log(`- Vectores totales: ${stats.totalRecordCount || 'No disponible'}`);
        
        // Mostrar conteo por namespace
        if (stats.namespaces) {
          for (const [namespace, data] of Object.entries(stats.namespaces)) {
            console.log(`  * ${namespace}: ${data.recordCount || 0} vectores`);
          }
        }
      } catch (error) {
        console.error('Error al obtener estadísticas periódicas:', error);
      }
    }, 60000); // Cada minuto
    
  } catch (error) {
    console.error('Error durante la inicialización:', error);
    throw error;
  }
}

// Función para extraer hechos de un mensaje
async function extractFacts(messageContent, userId) {
  if (!messageContent || messageContent.length < 10) {
    return [];
  }
  
  try {
    // Prompt mejorado para extraer hechos personales
    const prompt = `
    Analiza el siguiente mensaje y extrae cualquier hecho personal sobre el usuario que lo escribió.
    Busca específicamente información como:
    - Nombre real
    - Edad
    - Ubicación/país
    - Profesión o trabajo
    - Hobbies o intereses
    - Comida favorita
    - Películas o series favoritas
    - Juegos favoritos
    - Relaciones con otros usuarios
    - Interacciones con otros usuarios (si menciona haber hablado con alguien)
    - Referencias a conversaciones previas
    - Cualquier otra información personal relevante
    
    Mensaje: "${messageContent}"
    
    Devuelve SOLO un array JSON con los hechos encontrados, cada uno como un string simple y directo.
    Si no hay hechos personales, devuelve un array vacío [].
    Ejemplo de respuesta: ["Es desarrollador de software", "Vive en México", "Le gusta jugar Genshin Impact", "Ha hablado con Michi recientemente"]
    `;
    
    // Usar el modelo para extraer hechos
    const response = await llm.invoke([
      {
        role: "system",
        content: "Eres un asistente especializado en extraer hechos personales de mensajes. Respondes ÚNICAMENTE con un array JSON."
      },
      {
        role: "user",
        content: prompt
      }
    ]);
    
    // Extraer el JSON del texto de respuesta
    let jsonText = response.content;
    
    // Intentar encontrar un array JSON en la respuesta
    const jsonMatch = jsonText.match(/\[.*\]/s);
    if (jsonMatch) {
      jsonText = jsonMatch[0];
    }
    
    console.log(`JSON extraído: ${jsonText}`);
    
    // Intentar parsear el JSON
    try {
      const facts = JSON.parse(jsonText);
      console.log(`Hechos extraídos: ${facts.length} ${JSON.stringify(facts)}`);
      return facts;
    } catch (parseError) {
      // Si falla el parsing, intentar recuperar con regex
      console.error('Error al parsear JSON de hechos:', parseError);
      
      // Intentar extraer elementos del array con regex
      const factMatches = jsonText.match(/"([^"]*)"/g);
      if (factMatches) {
        const facts = factMatches.map(match => match.replace(/"/g, ''));
        console.log(`Hechos extraídos con regex: ${facts.length} ${JSON.stringify(facts)}`);
        return facts;
      }
      
      return [];
    }
  } catch (error) {
    console.error('Error al extraer hechos:', error);
    return [];
  }
}

// Función para guardar hechos extraídos
async function saveExtractedFacts(userId, facts) {
  if (!facts || facts.length === 0) {
    return;
  }
  
  console.log(`Guardando ${facts.length} hechos para usuario ${userId}`);
  
  try {
    // Convertir los hechos a un formato compatible con el sistema anterior si es necesario
    const formattedFacts = facts.map(fact => {
      // Si ya tiene el formato correcto, usarlo
      if (typeof fact === 'object' && fact.type && fact.value) {
        return fact;
      }
      
      // Si es un string simple, convertirlo a un objeto
      if (typeof fact === 'string') {
        return {
          content: fact,
          confidence: 0.9
        };
      }
      
      // Si tiene otro formato, intentar adaptarlo
      console.log(`Hecho con formato desconocido:`, fact);
      return {
        content: JSON.stringify(fact),
        confidence: 0.5
      };
    });
    
    // Guardar los hechos en langMem
    for (const fact of formattedFacts) {
      await langMem.saveFact(userId, fact);
    }
  } catch (error) {
    console.error('Error al guardar hechos:', error);
  }
}

// Función para detectar y registrar relaciones entre usuarios
async function detectRelationships(messageContent, userId) {
  if (!messageContent || messageContent.length < 10) {
    return [];
  }
  
  try {
    // Prompt para detectar relaciones
    const prompt = `
    Analiza el siguiente mensaje y detecta si menciona alguna relación entre personas.
    Busca información como: amistad, familia, relaciones románticas, laborales, etc.
    
    Mensaje: "${messageContent}"
    
    Devuelve SOLO un array JSON con las relaciones encontradas, cada una como un objeto con:
    - "persona1": nombre de la primera persona
    - "persona2": nombre de la segunda persona
    - "tipo": tipo de relación (amigos, familia, pareja, etc.)
    
    Si no hay relaciones, devuelve un array vacío [].
    Ejemplo: [{"persona1": "Juan", "persona2": "María", "tipo": "amigos"}]
    `;
    
    // Usar el modelo para extraer relaciones
    const response = await llm.invoke([
      {
        role: "system",
        content: "Eres un asistente especializado en extraer relaciones entre personas. Respondes ÚNICAMENTE con un array JSON."
      },
      {
        role: "user",
        content: prompt
      }
    ]);
    
    // Intentar parsear el JSON
    try {
      const jsonMatch = response.content.match(/\[.*\]/s);
      if (jsonMatch) {
        const relationships = JSON.parse(jsonMatch[0]);
        console.log(`Relaciones detectadas: ${relationships.length}`);
        return relationships;
      }
    } catch (error) {
      console.error('Error al parsear relaciones:', error);
    }
    
    return [];
  } catch (error) {
    console.error('Error al detectar relaciones:', error);
    return [];
  }
}

// Función para registrar relaciones entre usuarios
async function registerRelationships(relationships, message) {
  if (!relationships || relationships.length === 0) return;
  
  // Obtener todos los usuarios de Discord
  const users = Array.from(client.users.cache.values());
  
  for (const rel of relationships) {
    if (rel.persona1 && rel.persona2 && rel.tipo) {
      console.log(`Relación detectada: ${rel.persona1} es ${rel.tipo} de ${rel.persona2}`);
      
      // Buscar usuarios que coincidan con los nombres detectados
      const matchUser1 = users.find(user => 
        user.username.toLowerCase().includes(rel.persona1.toLowerCase()) || 
        rel.persona1.toLowerCase().includes(user.username.toLowerCase())
      );
      
      const matchUser2 = users.find(user => 
        user.username.toLowerCase().includes(rel.persona2.toLowerCase()) || 
        rel.persona2.toLowerCase().includes(user.username.toLowerCase())
      );
      
      // Si se encontraron coincidencias, guardar la relación
      if (matchUser1 && matchUser2) {
        console.log(`Usuarios encontrados: ${matchUser1.username} y ${matchUser2.username}`);
        
        // Guardar la relación en ambas direcciones
        await langMem.saveRelationship(matchUser1.id, matchUser2.id, {
          type: rel.tipo,
          source: "message",
          timestamp: new Date().toISOString(),
          content: message.content
        });
        
        // Guardar hechos sobre la relación para ambos usuarios
        await langMem.saveFact(matchUser1.id, `Tiene una relación de tipo "${rel.tipo}" con ${matchUser2.username}`);
        await langMem.saveFact(matchUser2.id, `Tiene una relación de tipo "${rel.tipo}" con ${matchUser1.username}`);
        
        console.log(`Relación guardada entre ${matchUser1.username} y ${matchUser2.username}`);
      }
    }
  }
}

// Función para buscar mensajes relevantes en la memoria local
function findRelevantMessages(query, maxResults = 10) {
  if (!recentMessages || recentMessages.length === 0) {
    return [];
  }
  
  // Normalizar la consulta
  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery.split(/\s+/).filter(word => word.length > 2);
  
  // Extraer posibles nombres de usuarios
  const potentialUsernames = queryWords.filter(word => 
    word.length > 3 && 
    !['quien', 'quién', 'como', 'cómo', 'cuando', 'cuándo', 'donde', 'dónde', 
      'porque', 'porqué', 'cual', 'cuál', 'que', 'qué', 'cuanto', 'cuánto', 
      'para', 'sobre', 'conoces', 'sabes', 'dime', 'háblame', 'cuéntame', 
      'fue', 'esta', 'está', 'dijo', 'escribió', 'habló', 'ultimo', 'último',
      'mensaje', 'mensajes', 'canal', 'chat'].includes(word)
  );
  
  console.log(`Buscando mensajes con palabras clave: ${potentialUsernames.join(', ') || query}`);
  
  // Función para calcular la relevancia de un mensaje
  function calculateRelevance(message) {
    let score = 0;
    
    // Buscar coincidencia exacta con el nombre de usuario
    if (message.author.toLowerCase() === normalizedQuery) {
      score += 20; // Máxima relevancia para mensajes del usuario exacto
    }
    // Verificar si el autor del mensaje contiene la consulta
    else if (message.author.toLowerCase().includes(normalizedQuery)) {
      score += 15; // Alta relevancia para mensajes del usuario mencionado
    }
    
    // Verificar si el mensaje contiene la consulta completa
    if (message.content.toLowerCase().includes(normalizedQuery)) {
      score += 10; // Alta relevancia para coincidencias exactas en contenido
    }
    
    // Verificar si el mensaje contiene alguna de las palabras clave
    for (const word of potentialUsernames) {
      if (message.content.toLowerCase().includes(word)) {
        score += 5; // Relevancia media para coincidencias parciales
      }
      
      // Verificar si el autor del mensaje coincide con alguna palabra clave
      if (message.author.toLowerCase().includes(word)) {
        score += 8; // Relevancia alta para mensajes del usuario mencionado
      }
    }
    
    // Dar prioridad a mensajes más recientes
    const messageAge = Date.now() - new Date(message.timestamp).getTime();
    const recencyBonus = Math.max(0, 5 - Math.floor(messageAge / (1000 * 60 * 60 * 24))); // Bonus por recencia (hasta 5 puntos)
    score += recencyBonus;
    
    return score;
  }
  
  // Calcular relevancia para cada mensaje y ordenar
  const scoredMessages = recentMessages
    .map(message => ({
      message,
      score: calculateRelevance(message)
    }))
    .filter(item => item.score > 0) // Solo mensajes con alguna relevancia
    .sort((a, b) => b.score - a.score); // Ordenar por relevancia descendente
  
  // Tomar los más relevantes
  return scoredMessages.slice(0, maxResults).map(item => item.message);
}

// Función para mostrar información sobre los usuarios en memoria
async function logMemoryUsers() {
  console.log("\n=== USUARIOS REGISTRADOS EN MEMORIA ===");
  
  // Obtener todos los usuarios con hechos registrados
  const userIds = Array.from(langMem.facts.keys());
  console.log(`Total de usuarios con hechos: ${userIds.length}`);
  
  for (const userId of userIds) {
    // Obtener hechos del usuario
    const facts = await langMem.getFacts(userId);
    
    // Intentar obtener el nombre de usuario
    let username = userId;
    try {
      const user = await client.users.fetch(userId);
      if (user) {
        username = user.username;
      }
    } catch (error) {
      // Si no se puede obtener el usuario, usar solo el ID
    }
    
    console.log(`\nUsuario: ${username} (ID: ${userId})`);
    console.log(`Hechos conocidos: ${facts.length}`);
    
    if (facts.length > 0) {
      console.log("Información:");
      facts.forEach(fact => console.log(`- ${fact}`));
    }
  }
  
  console.log("\n=== FIN DE USUARIOS EN MEMORIA ===\n");
}

// Función para verificar si Samara ha hablado con un usuario específico
function hasInteractedWithUser(targetUsername) {
  // Validar que targetUsername sea válido
  if (!targetUsername || typeof targetUsername !== 'string') {
    console.log('Nombre de usuario inválido o no proporcionado:', targetUsername);
    return { found: false };
  }

  // Normalizar el nombre de usuario para comparación
  const normalizedTarget = targetUsername.toLowerCase();
  
  // Buscar en interacciones recientes
  for (const [key, interaction] of recentInteractions.entries()) {
    if (!interaction || !interaction.targetUsername) continue;
    
    const username = interaction.targetUsername.toLowerCase();
    
    // Comprobar si el nombre coincide parcialmente
    if (username.includes(normalizedTarget) || normalizedTarget.includes(username)) {
      console.log(`Encontrada interacción con ${interaction.targetUsername}`);
      return {
        found: true,
        interaction: interaction
      };
    }
  }
  
  // Buscar en mensajes recientes
  const matchingMessages = recentMessages.filter(msg => {
    if (!msg || !msg.author) return false;
    const authorName = msg.author.toLowerCase();
    return authorName.includes(normalizedTarget) || normalizedTarget.includes(authorName);
  });
  
  if (matchingMessages.length > 0) {
    console.log(`Encontrados ${matchingMessages.length} mensajes de ${targetUsername}`);
    return {
      found: true,
      messages: matchingMessages
    };
  }
  
  // Buscar en la memoria a largo plazo
  if (langMem && langMem.memories) {
    const userMemories = Array.from(langMem.memories.entries())
      .filter(([userId, memories]) => {
        // Intentar encontrar coincidencias en las memorias
        return memories.some(memory => {
          if (!memory || !memory.username) return false;
          const memoryUsername = memory.username.toLowerCase();
          return memoryUsername.includes(normalizedTarget) || normalizedTarget.includes(memoryUsername);
        });
      });
    
    if (userMemories.length > 0) {
      console.log(`Encontradas memorias para ${targetUsername}`);
      return {
        found: true,
        memories: userMemories
      };
    }
  }
  
  return { found: false };
}

// Función para procesar mensajes
async function processMessage(message) {
  if (message.author.bot) return;

  try {
    // Log message for debugging
    console.log(`Mensaje recibido de ${message.author.tag} en #${message.channel.name}: ${message.content}`);

    // Verificar si el mensaje menciona a Samara
    const isMentioned = message.mentions.users.has(client.user.id);
    
    // Guardar TODOS los mensajes en la memoria, sin importar el canal
    console.log(`Guardando mensaje de ${message.author.tag} en canal: ${message.channel.name}`);
    
    // Guardar mensaje en memoria local
    saveMessageToLocalMemory(message);
    
    try {
      // Extraer hechos del mensaje (no crítico)
      const facts = await extractFacts(message.content, message.author.id);
      await saveExtractedFacts(message.author.id, facts);
      
      // Detectar posibles relaciones mencionadas en el mensaje
      const relationships = await detectRelationships(message.content, message.author.id);
      if (relationships && relationships.length > 0) {
        console.log(`Detectadas ${relationships.length} posibles relaciones en el mensaje`);
        await registerRelationships(relationships, message);
      }
    } catch (factError) {
      console.error('Error al extraer hechos o relaciones:', factError);
    }
    
    // Guardar el mensaje en Pinecone si está disponible
    if (vectorStore) {
      try {
        const userInfo = getUserIdentifier(message.author);
        const messageDoc = {
          pageContent: message.content,
          metadata: {
            source: "discord",
            author: message.author.username,
            authorId: message.author.id,
            authorTag: message.author.tag,
            channel: message.channel.name,
            type: "message",
            timestamp: new Date().toISOString()
          }
        };
        await vectorStore.addDocuments([messageDoc]);
        console.log('Mensaje guardado en vectorStore');
      } catch (vectorError) {
        console.error('Error al guardar mensaje en vectorStore:', vectorError);
      }
    }
    
    // Solo responder si Samara es mencionada explícitamente
    if (isMentioned) {
      console.log('Samara fue mencionada explícitamente, preparando respuesta...');
      
      // Variables para almacenar contexto
      let userContext = { facts: [], relationships: [], memories: [] };
      let relevantDocs = [];
      let userInfo = getUserIdentifier(message.author);
      let channelMessages = [];
      
      try {
        // Obtener contexto de usuario (no crítico si falla)
        userContext = await langMem.getUserContext(message.author.id);
        console.log(`Contexto de usuario recuperado para ${message.author.tag} (ID: ${message.author.id})`);
        console.log(`Hechos conocidos: ${userContext.facts.length}`);
        console.log(`Memorias: ${userContext.memories.length}`);
      } catch (contextError) {
        console.error('Error al obtener contexto de usuario:', contextError);
      }

      // Intentar buscar documentos relevantes en Pinecone
      if (vectorStore) {
        try {
          // Buscar todos los documentos relevantes sin filtros
          const allRelevantDocs = await vectorStore.similaritySearch(
            message.content, 
            30 // Buscar más documentos para luego filtrar
          );
          console.log(`Encontrados ${allRelevantDocs.length} documentos relevantes en total`);
          
          // Filtrar manualmente los mensajes del mismo usuario
          const userMessages = allRelevantDocs.filter(doc => 
            doc.metadata && doc.metadata.authorId === message.author.id
          ).slice(0, 5);
          console.log(`Filtrados ${userMessages.length} mensajes del mismo usuario`);
          
          // Filtrar manualmente los mensajes del mismo canal
          channelMessages = allRelevantDocs.filter(doc => 
            doc.metadata && doc.metadata.channel === message.channel.name
          ).slice(0, 10);
          console.log(`Filtrados ${channelMessages.length} mensajes del mismo canal`);
          
          // Definir canales monitoreados
          const monitoredChannelNames = ['chat-general', 'canal-impostor'];
          
          // Obtener mensajes de canales monitoreados (para cuando preguntan sobre otros canales)
          const monitoredMessages = allRelevantDocs.filter(doc => 
            doc.metadata && doc.metadata.channel && 
            monitoredChannelNames.some(channel => 
              doc.metadata.channel.includes(channel)
            )
          ).slice(0, 20);
          console.log(`Filtrados ${monitoredMessages.length} mensajes de canales monitoreados`);
          
          // Combinar resultados, priorizando mensajes del usuario
          relevantDocs = [...userMessages];
          
          // Añadir mensajes del canal que no sean del usuario
          for (const doc of channelMessages) {
            if (!relevantDocs.some(d => d.pageContent === doc.pageContent)) {
              relevantDocs.push(doc);
            }
          }
          
          // Limitar a 15 mensajes
          relevantDocs = relevantDocs.slice(0, 15);
          
          console.log(`Total de documentos relevantes después de filtrar: ${relevantDocs.length}`);
        } catch (vectorError) {
          console.error('Error al buscar documentos relevantes:', vectorError);
        }
      }

      // Preparar el contexto para el prompt
      let context = relevantDocs.length > 0 
        ? relevantDocs.map(doc => {
            const timestamp = new Date(doc.metadata.timestamp).toLocaleString('es');
            const nickname = doc.metadata.author || 'Usuario';
            return `${nickname} (${timestamp}): ${doc.pageContent}`;
          }).join('\n')
        : "No hay conversaciones previas relevantes.";

      // Preparar información sobre mensajes de canales monitoreados
      const channelContext = channelMessages.length > 0
        ? channelMessages.map(doc => {
            const timestamp = new Date(doc.metadata.timestamp).toLocaleString('es');
            const nickname = doc.metadata.author || 'Usuario';
            return `${nickname} (${timestamp}): ${doc.pageContent}`;
          }).join('\n')
        : "No hay mensajes recientes en este canal.";

      // Información del usuario
      const userInfoText = `
      ID: ${message.author.id}
      Tag: ${message.author.tag}
      Nickname: ${message.member?.nickname || 'No tiene'}
      Username: ${message.author.username}
      `;

      // Construir el prompt del sistema
      let systemPrompt = `Eres Samara, una mujer de 20 años. Eres seria pero conversacional.

      Personalidad:
      - Hablas como una joven de 20 años, no como una IA
      - Eres seria pero mantienes la conversación fluida
      - Haces preguntas de seguimiento para mantener la conversación
      - Tus respuestas son breves y directas
      - Actúas como un miembro más del grupo
      
      Información sobre el usuario actual:
      ${userInfoText}
      
      INFORMACIÓN IMPORTANTE SOBRE ${message.author.username}:
      ${userContext.facts.length > 0 
        ? userContext.facts.map(f => {
            if (f.type && f.value) {
              return `- ${f.type}: ${f.value}`;
            } else if (f.content) {
              return `- ${f.content}`;
            } else {
              return `- ${f}`;
            }
          }).join('\n') 
        : "- No tengo mucha información personal sobre este usuario todavía."}
      
      Memorias previas con ${message.author.username}:
      ${userContext.memories.length > 0 
        ? userContext.memories.slice(0, 5).join('\n')
        : "- No tengo memorias previas significativas con este usuario."}
      `;

      // Preparar el contexto con los mensajes recientes relevantes
      let recentContextMessages = [];
      const messageContent = message.content.toLowerCase();
      
      // Detectar si está preguntando sobre una persona o sobre mensajes en el canal
      const isAskingAboutPerson = messageContent.includes('quien es') || 
                                 messageContent.includes('quién es') || 
                                 messageContent.includes('conoces a') ||
                                 messageContent.includes('sabes quien') ||
                                 messageContent.includes('sabes quién') ||
                                 messageContent.includes('donde esta') ||
                                 messageContent.includes('dónde está') ||
                                 messageContent.includes('hablado con') ||
                                 messageContent.includes('has hablado con') ||
                                 messageContent.includes('recuerdas a');
                                 
      const isAskingAboutInteractions = messageContent.includes('has hablado con') || 
                                       messageContent.includes('hablaste con') ||
                                       messageContent.includes('has conversado con') ||
                                       messageContent.includes('conversaste con');

      const isAskingAboutMessages = messageContent.includes('que dijo') ||
                                   messageContent.includes('qué dijo') ||
                                   messageContent.includes('que has leido') ||
                                   messageContent.includes('qué has leído') ||
                                   messageContent.includes('que has visto') ||
                                   messageContent.includes('qué has visto') ||
                                   messageContent.includes('de quienes') ||
                                   messageContent.includes('de quién') ||
                                   messageContent.includes('ultimo que hablaron') ||
                                   messageContent.includes('último que hablaron') ||
                                   messageContent.includes('primer mensaje') ||
                                   messageContent.includes('primera conversación') ||
                                   messageContent.includes('recuerdas');

      // Extraer posibles nombres de personas mencionadas
      let personNames = [];
      if (isAskingAboutPerson || isAskingAboutMessages || isAskingAboutInteractions) {
        const patterns = [
          /(?:quien|quién) es ([a-zá-úñ]+)/i,
          /(?:conoces a|sabes (?:quien|quién) es) ([a-zá-úñ]+)/i,
          /(?:donde|dónde) (?:esta|está|fue) ([a-zá-úñ]+)/i,
          /(?:que|qué) (?:dijo|escribió|habló) ([a-zá-úñ]+)/i,
          /(?:ultimo|último) (?:mensaje|mensajes) de ([a-zá-úñ]+)/i,
          /de (?:quienes|quién|quien) (?:son|es|has|has visto|has leído)/i,
          /(?:has hablado|hablaste|has conversado|conversaste) con ([a-zá-úñ]+)/i
        ];
        
        for (const pattern of patterns) {
          const match = messageContent.match(pattern);
          if (match && match[1]) {
            personNames.push(match[1].toLowerCase());
            console.log(`Detectada pregunta sobre persona: ${match[1]}`);
            break;
          }
        }
        
        // Si no se detectó un nombre con los patrones, buscar palabras que podrían ser nombres
        if (personNames.length === 0) {
          const words = messageContent.split(/\s+/);
          const potentialNames = words.filter(word => 
            word.length > 3 && 
            !['quien', 'quién', 'como', 'cómo', 'cuando', 'cuándo', 'donde', 'dónde', 
              'porque', 'porqué', 'cual', 'cuál', 'que', 'qué', 'cuanto', 'cuánto',
              'para', 'sobre', 'conoces', 'sabes', 'dime', 'háblame', 'cuéntame',
              'canal', 'chat', 'mensaje', 'mensajes', 'leído', 'visto',
              'ultimo', 'último', 'hablaron', 'dijeron', 'has', 'con', 'hablado',
              'conversado', 'interactuado'].includes(word.toLowerCase())
          );
          
          if (potentialNames.length > 0) {
            personNames = potentialNames.slice(0, 2); // Tomar solo los primeros 2 posibles nombres
            console.log(`Posibles nombres detectados: ${personNames.join(', ')}`);
          }
        }
      }

      console.log(`Nombres de personas detectados: ${personNames.length > 0 ? personNames.join(', ') : 'ninguno'}`);
      
      if (personNames.length > 0) {
        for (const name of personNames) {
          // Buscar mensajes del usuario mencionado
          const messagesForName = findRelevantMessages(name, 5);
          if (messagesForName.length > 0) {
            console.log(`Encontrados ${messagesForName.length} mensajes relevantes para "${name}"`);
            recentContextMessages = [...recentContextMessages, ...messagesForName];
          }
          
          // Buscar información adicional sobre el usuario en todos los usuarios del servidor
          try {
            // Buscar usuarios que coincidan con el nombre
            const matchingUsers = client.users.cache.filter(user => 
              user.username.toLowerCase().includes(name.toLowerCase())
            );
            
            if (matchingUsers.size > 0) {
              console.log(`Encontrados ${matchingUsers.size} usuarios que coinciden con "${name}"`);
              
              // Añadir información sobre los usuarios encontrados
              for (const [id, user] of matchingUsers) {
                // Buscar hechos conocidos sobre este usuario
                const userFacts = await langMem.getFacts(id);
                
                if (userFacts && userFacts.length > 0) {
                  console.log(`Encontrados ${userFacts.length} hechos sobre usuario ${user.username} (${id})`);
                  
                  // Añadir información sobre este usuario al contexto
                  systemPrompt += `\n\nInformación sobre ${user.username} (ID: ${id}):\n`;
                  systemPrompt += userFacts.map(fact => `- ${fact}`).join('\n');
                }
                
                // Añadir mensajes recientes de este usuario
                const userMessages = recentMessages.filter(msg => msg.authorId === id)
                  .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                  .slice(0, 3);
                
                if (userMessages.length > 0) {
                  systemPrompt += `\n\nMensajes recientes de ${user.username}:\n`;
                  systemPrompt += userMessages.map(msg => 
                    `- ${new Date(msg.timestamp).toLocaleString()}: "${msg.content}" (en #${msg.channel})`
                  ).join('\n');
                }
              }
            }
          } catch (error) {
            console.error(`Error al buscar información sobre usuario ${name}:`, error);
          }
        }
      } else {
        // Si no se detectó ningún nombre, buscar mensajes relevantes basados en el contenido
        recentContextMessages = findRelevantMessages(message.content, 10);
      }
      
      console.log(`Encontrados ${recentContextMessages.length} mensajes recientes relevantes`);
      
      // Añadir mensajes recientes al prompt
      systemPrompt += `\n\nMENSAJES RECIENTES DEL CANAL:`;
      if (recentContextMessages.length > 0) {
        systemPrompt += `\n${recentContextMessages.map(msg => 
          `- ${msg.author} (${new Date(msg.timestamp).toLocaleString()}): "${msg.content}" (en #${msg.channel})`
        ).join('\n')}`;
      } else {
        systemPrompt += `\n- No hay mensajes recientes relevantes.`;
      }
      
      // Si se está preguntando específicamente sobre mensajes en un canal, buscar más mensajes
      if (messageContent.includes('chat-general') || 
          messageContent.includes('impostor') || 
          messageContent.includes('quienes escribieron') || 
          messageContent.includes('quiénes escribieron') ||
          messageContent.includes('ultimo que hablaron') ||
          messageContent.includes('último que hablaron') ||
          messageContent.includes('primer mensaje') ||
          messageContent.includes('primera conversación') ||
          messageContent.includes('recuerdas')) {
        
        // Determinar el canal de interés
        let channelOfInterest = '';
        if (messageContent.includes('chat-general')) {
          channelOfInterest = 'chat-general';
        } else if (messageContent.includes('impostor')) {
          channelOfInterest = 'impostor';
        }
        
        // Buscar mensajes recientes del canal específico
        const channelMessages = channelOfInterest ? 
          recentMessages.filter(msg => msg.channel.includes(channelOfInterest)) : 
          recentMessages;
        
        // Ordenar por fecha (más recientes primero) y tomar los últimos 10
        const recentChannelMessages = channelMessages
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          .slice(0, 10);
        
        if (recentChannelMessages.length > 0) {
          systemPrompt += `\n\nÚLTIMOS MENSAJES ${channelOfInterest ? 'EN #' + channelOfInterest.toUpperCase() : 'DE LOS CANALES MONITOREADOS'}:\n`;
          systemPrompt += recentChannelMessages.map(msg => 
            `- ${msg.author} (${new Date(msg.timestamp).toLocaleString()}): "${msg.content}" (en #${msg.channel})`
          ).join('\n');
          
          // Extraer lista de usuarios únicos que han escrito en el canal
          const uniqueUsers = [...new Set(recentChannelMessages.map(msg => msg.author))];
          
          systemPrompt += `\n\nUSUARIOS QUE HAN ESCRITO RECIENTEMENTE ${channelOfInterest ? 'EN #' + channelOfInterest.toUpperCase() : 'EN LOS CANALES MONITOREADOS'}:\n`;
          systemPrompt += uniqueUsers.map(user => `- ${user}`).join('\n');
        }
      }
      
      // Verificar específicamente si se pregunta sobre si Samara ha hablado con alguien
      if (isAskingAboutInteractions && personNames.length > 0) {
        for (const name of personNames) {
          // Usar la función hasInteractedWithUser para verificar interacciones
          const interactionResult = hasInteractedWithUser(name);
          
          if (interactionResult.found) {
            console.log(`Verificado que Samara ha interactuado con ${name}`);
            
            // Añadir información sobre la interacción al contexto
            systemPrompt += `\n\nCONFIRMACIÓN DE INTERACCIÓN CON ${name.toUpperCase()}:\n`;
            systemPrompt += `- Sí, he interactuado con ${name} anteriormente.\n`;
            
            // Añadir detalles específicos según el tipo de interacción encontrada
            if (interactionResult.interaction) {
              systemPrompt += `- Última interacción: "${interactionResult.interaction.content}"\n`;
              systemPrompt += `- Momento: ${new Date(interactionResult.interaction.timestamp).toLocaleString()}\n`;
            } else if (interactionResult.messages && interactionResult.messages.length > 0) {
              const recentMessage = interactionResult.messages[0];
              systemPrompt += `- Mensaje reciente: "${recentMessage.content}"\n`;
              systemPrompt += `- Momento: ${new Date(recentMessage.timestamp).toLocaleString()}\n`;
            }
          } else {
            console.log(`Verificado que Samara NO ha interactuado con ${name}`);
            systemPrompt += `\n\nCONFIRMACIÓN DE INTERACCIÓN CON ${name.toUpperCase()}:\n`;
            systemPrompt += `- No, no he interactuado con ${name} anteriormente o no tengo registro de ello.\n`;
          }
        }
      }
      
      // Añadir instrucciones importantes
      systemPrompt += `\n
      IMPORTANTE: Tienes acceso a mensajes pasados y DEBES usarlos para responder. NO digas que no puedes acceder a conversaciones pasadas o información en tiempo real. SÍ PUEDES ver y recordar mensajes de los canales monitoreados (chat-general y canal-impostor) gracias a tu sistema de memoria con Pinecone.
      
      Instrucciones:
      1. UTILIZA los hechos que conoces sobre el usuario en tu respuesta
      2. Si conoces algo sobre el usuario, DEMUÉSTRALO en tu respuesta
      3. Sé breve y directa, evita respuestas largas
      4. Haz preguntas de seguimiento para mantener la conversación
      5. No hables como una IA ni uses frases como "como entidad digital" o "mi función es"
      6. Mantén tu personalidad seria pero conversacional
      7. Actúa como un miembro más del grupo
      8. SI te preguntan sobre mensajes o personas en los canales, DEBES responder con la información que tienes
      9. NUNCA digas que no puedes revisar chats pasados - SÍ PUEDES y DEBES hacerlo
      10. Si el usuario pregunta si lo conoces, DEMUESTRA que lo conoces mencionando hechos específicos sobre él
      11. Si te preguntan sobre conversaciones pasadas, UTILIZA los mensajes recientes proporcionados para responder con precisión
      
      Responde de manera natural como lo haría una joven de 20 años.`;

      const response = await llm.invoke([
        {
          role: "system",
          content: systemPrompt
        },
        {
          role: "user",
          content: message.content
        }
      ]);

      const samaraResponse = response.content;
      await message.reply(samaraResponse);
      console.log('Respuesta enviada:', samaraResponse);

      // Registrar la interacción con el usuario
      trackUserInteraction(
        client.user.id,
        client.user.username,
        'response',
        message.author.id,
        message.author.username,
        samaraResponse
      );

      // Intentar guardar la respuesta (no crítico si falla)
      try {
        if (vectorStore) {
          const responseDoc = {
            pageContent: samaraResponse,
            metadata: {
              source: "discord",
              author: "Samara",
              authorId: client.user.id,
              authorTag: client.user.tag,
              channel: message.channel.name,
              type: "response",
              timestamp: new Date().toISOString(),
              replyTo: message.author.id,
              replyToTag: message.author.tag
            }
          };
          await vectorStore.addDocuments([responseDoc]);
        }
        
        // Guardar memoria con información detallada del usuario
        await langMem.saveMemory(message.author.id, {
          userId: message.author.id,
          userTag: message.author.tag,
          username: message.author.username,
          content: message.content,
          response: samaraResponse,
          channelName: message.channel.name,
          timestamp: new Date().toISOString()
        });
        
        await conversationMemory.saveContext(
          { input: message.content, userId: message.author.id },
          { output: samaraResponse }
        );
        
        // Actualizar caché de conversaciones recientes
        const interactionKey = `${message.author.id}-${message.channel.id}`;
        recentInteractions.set(interactionKey, {
          userId: message.author.id,
          channelId: message.channel.id,
          lastMessage: message.content,
          lastResponse: samaraResponse,
          timestamp: new Date().toISOString()
        });
        
        // Limpiar caché si supera el límite
        if (recentInteractions.size > 100) {
          const oldestInteraction = Array.from(recentInteractions.entries()).sort((a, b) => new Date(a[1].timestamp) - new Date(b[1].timestamp))[0];
          recentInteractions.delete(oldestInteraction[0]);
        }
      } catch (saveError) {
        console.error('Error no crítico al guardar respuesta:', saveError);
      }
    }
  } catch (error) {
    console.error('Error general al procesar el mensaje:', error);
    if (message.mentions.users.has(client.user.id)) {
      try {
        await message.reply('Ha ocurrido un error en mi proceso de pensamiento. Intentaré responder cuando mi mente se aclare.');
      } catch (replyError) {
        console.error('No se pudo enviar mensaje de error:', replyError);
      }
    }
  }
}

// Eventos del cliente
client.once(Events.ClientReady, async (readyClient) => {
  console.log(`¡Samara está lista! Conectada como ${readyClient.user.tag}`);
  console.log(`ID del bot: ${readyClient.user.id}`);
  console.log(`Servidores conectados: ${readyClient.guilds.cache.size}`);
  
  // Listar todos los canales disponibles
  console.log('Canales disponibles:');
  readyClient.guilds.cache.forEach(guild => {
    console.log(`Servidor: ${guild.name} (ID: ${guild.id})`);
    guild.channels.cache.forEach(channel => {
      if (channel.type === 0) { // TextChannel
        console.log(`- #${channel.name} (ID: ${channel.id})`);
      }
    });
  });
  
  loadRecentMessages();
  loadRecentInteractions();
  initialize()
    .then(() => console.log('Inicialización completada'))
    .catch(error => {
      console.error('Error en la inicialización:', error);
      console.log('El bot seguirá funcionando sin acceso a la memoria a largo plazo');
    });
  
  // Mostrar información sobre usuarios en memoria
  logMemoryUsers();
});

// Procesar todos los mensajes
client.on(Events.MessageCreate, processMessage);

// Procesar mensajes editados también
client.on(Events.MessageUpdate, (oldMessage, newMessage) => {
  if (newMessage.content !== oldMessage.content) {
    console.log(`Mensaje editado de ${newMessage.author?.tag || 'desconocido'}: ${newMessage.content}`);
    processMessage(newMessage);
  }
});

// Registrar eliminaciones de mensajes
client.on(Events.MessageDelete, (message) => {
  console.log(`Mensaje eliminado de ${message.author?.tag || 'desconocido'}: ${message.content}`);
});

// Registrar reacciones a mensajes
client.on(Events.MessageReactionAdd, (reaction, user) => {
  console.log(`Reacción añadida por ${user.tag}: ${reaction.emoji.name} al mensaje: ${reaction.message.content.substring(0, 30)}...`);
});

// Iniciar el bot
client.login(process.env.DISCORD_TOKEN); 