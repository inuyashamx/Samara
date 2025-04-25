const { BaseMemory } = require("langchain/memory");
const fs = require('fs');
const path = require('path');

class LangMem extends BaseMemory {
  constructor(options = {}) {
    super();
    this.dataDir = options.dataDir || path.join(__dirname, 'data');
    this.memoriesFile = path.join(this.dataDir, 'memories.json');
    this.relationshipsFile = path.join(this.dataDir, 'relationships.json');
    this.factsFilePath = path.join(this.dataDir, 'facts.json');
    
    // Crear directorio de datos si no existe
    if (!fs.existsSync(this.dataDir)) {
      fs.mkdirSync(this.dataDir, { recursive: true });
      console.log(`Directorio de datos creado: ${this.dataDir}`);
    }
    
    // Cargar datos desde archivos o inicializar nuevos
    this.memories = this._loadFromFile(this.memoriesFile, new Map());
    this.relationships = this._loadFromFile(this.relationshipsFile, new Map());
    this.facts = this._loadFromFile(this.factsFilePath, new Map());
    
    console.log(`LangMem inicializado con ${this.memories.size} usuarios con memorias, ${this.relationships.size} relaciones y ${this.facts.size} usuarios con hechos.`);
  }
  
  // Método para cargar datos desde un archivo
  _loadFromFile(filePath, defaultValue) {
    try {
      if (fs.existsSync(filePath)) {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        
        // Convertir objeto a Map
        if (defaultValue instanceof Map) {
          const map = new Map();
          for (const [key, value] of Object.entries(data)) {
            // Para facts, necesitamos convertir el array a Set
            if (filePath === this.factsFilePath) {
              // Verificar si value es un array antes de usar map
              if (Array.isArray(value)) {
                map.set(key, new Set(value.map(item => typeof item === 'string' ? { content: item, timestamp: new Date().toISOString() } : item)));
              } else {
                // Si no es un array, convertirlo a un Set con un solo elemento
                map.set(key, new Set([typeof value === 'string' ? { content: value, timestamp: new Date().toISOString() } : value]));
              }
            } else {
              map.set(key, value);
            }
          }
          console.log(`Datos cargados desde ${filePath}: ${map.size} entradas`);
          return map;
        }
        
        console.log(`Datos cargados desde ${filePath}`);
        return data;
      }
    } catch (error) {
      console.error(`Error al cargar datos desde ${filePath}:`, error);
    }
    
    console.log(`No se encontraron datos en ${filePath}, inicializando con valores por defecto`);
    return defaultValue;
  }
  
  // Método para guardar datos en un archivo
  _saveToFile(filePath, data) {
    try {
      // Convertir Map a objeto para JSON
      let jsonData;
      if (data instanceof Map) {
        jsonData = {};
        for (const [key, value] of data.entries()) {
          // Para facts, necesitamos convertir el Set a array
          if (filePath === this.factsFilePath) {
            // Verificar que value sea un Set antes de usar Array.from
            if (value instanceof Set) {
              jsonData[key] = Array.from(value);
            } else {
              // Si no es un Set, guardarlo como está
              jsonData[key] = value;
            }
          } else {
            jsonData[key] = value;
          }
        }
      } else {
        jsonData = data;
      }
      
      fs.writeFileSync(filePath, JSON.stringify(jsonData, null, 2), 'utf8');
      console.log(`Datos guardados en ${filePath}`);
    } catch (error) {
      console.error(`Error al guardar datos en ${filePath}:`, error);
    }
  }

  // Método para guardar un recuerdo
  async saveMemory(userId, memory) {
    if (!this.memories.has(userId)) {
      this.memories.set(userId, []);
    }
    this.memories.get(userId).push({
      ...memory,
      timestamp: new Date().toISOString()
    });
    
    // Guardar en archivo
    this._saveToFile(this.memoriesFile, this.memories);
  }

  // Método para guardar una relación
  async saveRelationship(userId1, userId2, relationship) {
    const key = `${userId1}-${userId2}`;
    this.relationships.set(key, {
      ...relationship,
      timestamp: new Date().toISOString()
    });
    
    // Guardar en archivo
    this._saveToFile(this.relationshipsFile, this.relationships);
  }

  // Guardar un hecho sobre un usuario
  async saveFact(userId, fact) {
    if (!userId || !fact) return;
    
    // Inicializar el conjunto de hechos para el usuario si no existe
    if (!this.facts.has(userId)) {
      this.facts.set(userId, new Set());
    }
    
    const userFacts = this.facts.get(userId);
    
    // Determinar el contenido del hecho según su formato
    let factContent;
    if (fact.type && fact.value) {
      // Formato antiguo: {type, value}
      factContent = `${fact.type}: ${fact.value}`;
    } else if (fact.content) {
      // Formato nuevo: {content}
      factContent = fact.content;
    } else {
      // Formato desconocido, usar JSON
      factContent = JSON.stringify(fact);
    }
    
    // Añadir el hecho al conjunto (Set garantiza unicidad)
    userFacts.add(factContent);
    
    // Guardar los cambios en el archivo
    await this._saveToFile(this.factsFilePath, this._mapToObject(this.facts));
    
    return factContent;
  }

  // Método para obtener hechos sobre un usuario
  async getFacts(userId) {
    if (!userId || !this.facts.has(userId)) {
      return [];
    }
    
    const factsSet = this.facts.get(userId);
    if (!factsSet || factsSet.size === 0) {
      return [];
    }
    
    const facts = Array.from(factsSet);
    
    // Procesar los hechos según su formato
    const processedFacts = facts.map(fact => {
      // Si es un string, devolverlo directamente
      if (typeof fact === 'string') {
        return fact;
      }
      
      // Si es un objeto con content, usar ese valor
      if (fact.content) {
        return fact.content;
      }
      
      // Si es un objeto con type y value, formatear como "type: value"
      if (fact.type && fact.value) {
        return `${fact.type}: ${fact.value}`;
      }
      
      // Caso de último recurso, convertir a string
      return JSON.stringify(fact);
    });
    
    // Eliminar posibles duplicados
    return [...new Set(processedFacts)];
  }

  // Método para obtener un hecho específico de un usuario
  async getUserFact(userId, factType) {
    const facts = await this.getFacts(userId);
    return facts.find(f => f.type === factType || f.content === factType);
  }

  // Método para obtener todos los recuerdos de un usuario
  async getMemories(userId) {
    return this.memories.get(userId) || [];
  }

  // Método para obtener relaciones entre usuarios
  async getRelationship(userId1, userId2) {
    const key = `${userId1}-${userId2}`;
    return this.relationships.get(key);
  }

  // Método para obtener el contexto completo de un usuario
  async getUserContext(userId) {
    const memories = await this.getMemories(userId);
    const facts = await this.getFacts(userId);
    const relationships = Array.from(this.relationships.entries())
      .filter(([key]) => key.includes(userId))
      .map(([_, value]) => value);

    return {
      memories,
      facts,
      relationships
    };
  }

  // Implementación requerida por BaseMemory
  get memoryKeys() {
    return ["memories", "relationships", "facts"];
  }

  // Método para cargar memoria en el contexto
  async loadMemoryVariables(values) {
    const userId = values.userId;
    const context = await this.getUserContext(userId);
    return {
      memories: context.memories,
      relationships: context.relationships,
      facts: context.facts
    };
  }

  // Método para guardar contexto
  async saveContext(inputValues, outputValues) {
    const userId = inputValues.userId;
    if (userId) {
      await this.saveMemory(userId, {
        input: inputValues.input,
        output: outputValues.output
      });
    }
  }

  _mapToObject(map) {
    const obj = {};
    for (const [key, value] of map.entries()) {
      // Si el valor es un Set, convertirlo a array
      if (value instanceof Set) {
        obj[key] = Array.from(value);
      } else if (value instanceof Map) {
        // Si el valor es otro Map, convertirlo recursivamente
        obj[key] = this._mapToObject(value);
      } else {
        // Valor normal
        obj[key] = value;
      }
    }
    return obj;
  }
}

module.exports = LangMem;