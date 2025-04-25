const { ChatOpenAI } = require('@langchain/openai');
require('dotenv').config();

// Configuración de OpenAI
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4",
  temperature: 0.7
});

// Función para extraer hechos del mensaje
async function extractFacts(message) {
  const response = await llm.invoke([
    {
      role: "system",
      content: `Eres un asistente especializado en extraer hechos sobre usuarios a partir de mensajes.
      
      INSTRUCCIONES IMPORTANTES:
      1. Analiza el mensaje del usuario y extrae cualquier información personal relevante.
      2. Presta especial atención a: profesión, comida favorita, hobbies, ubicación, edad, etc.
      3. Devuelve ÚNICAMENTE un array JSON válido sin ningún texto adicional.
      4. Si no hay hechos relevantes, devuelve un array vacío: []
      5. NO incluyas explicaciones, comentarios o texto adicional antes o después del JSON.
      
      Formato requerido para cada hecho:
      {
        "type": "profesión", // Tipo de hecho (profesión, comida_favorita, hobby, etc.)
        "value": "ingeniero", // El valor específico
        "confidence": 0.9 // Nivel de confianza entre 0 y 1
      }
      
      Ejemplo de respuesta correcta para "Me gusta programar y mi comida favorita es la pizza":
      [{"type":"hobby","value":"programar","confidence":0.9},{"type":"comida_favorita","value":"pizza","confidence":0.9}]
      
      Ejemplo de respuesta correcta cuando no hay hechos:
      []`
    },
    {
      role: "user",
      content: message
    }
  ]);

  console.log("Respuesta completa del LLM:");
  console.log(response.content);

  try {
    // Intentar extraer solo el JSON de la respuesta
    let jsonText = response.content.trim();
    
    // Buscar el primer '[' y el último ']' para extraer solo el array JSON
    const startIndex = jsonText.indexOf('[');
    const endIndex = jsonText.lastIndexOf(']');
    
    if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
      jsonText = jsonText.substring(startIndex, endIndex + 1);
      console.log('JSON extraído:', jsonText);
      
      const facts = JSON.parse(jsonText);
      console.log(`\nHechos extraídos (${facts.length}):`);
      console.log(JSON.stringify(facts, null, 2));
      return facts;
    } else {
      console.log('No se encontró un array JSON válido en la respuesta');
      return [];
    }
  } catch (error) {
    console.error('Error al parsear hechos:', error);
    
    // Intento de recuperación: buscar cualquier array en la respuesta
    try {
      const match = response.content.match(/\[.*\]/s);
      if (match) {
        const jsonText = match[0];
        console.log('Intento de recuperación con regex:', jsonText);
        const facts = JSON.parse(jsonText);
        console.log(`\nHechos extraídos con regex (${facts.length}):`);
        console.log(JSON.stringify(facts, null, 2));
        return facts;
      }
    } catch (regexError) {
      console.error('Error en recuperación con regex:', regexError);
    }
    
    return [];
  }
}

// Mensajes de prueba
const testMessages = [
  "Soy programador y me encanta la pizza. Tengo 30 años y vivo en México.",
  "Mi profesión es médico y mi comida favorita es el sushi.",
  "Me gusta jugar videojuegos en mi tiempo libre.",
  "Hola, ¿cómo estás?",
  "Trabajo como ingeniero y vivo en España. Me gusta el chocolate."
];

// Función principal para probar la extracción de hechos
async function runTests() {
  console.log("=== PRUEBA DE EXTRACCIÓN DE HECHOS ===\n");
  
  for (let i = 0; i < testMessages.length; i++) {
    console.log(`\n--- PRUEBA ${i+1}: "${testMessages[i]}" ---\n`);
    await extractFacts(testMessages[i]);
  }
  
  console.log("\n=== FIN DE PRUEBAS ===");
}

// Ejecutar las pruebas
runTests().catch(console.error);
