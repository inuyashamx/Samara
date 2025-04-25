const { PineconeStore } = require('@langchain/pinecone');
const { OpenAIEmbeddings } = require('@langchain/openai');

class SamaraMemory {
  constructor(pineconeClient) {
    this.pineconeClient = pineconeClient;
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
  }

  async initialize() {
    const index = this.pineconeClient.index(process.env.PINECONE_INDEX);
    this.vectorStore = await PineconeStore.fromExistingIndex(
      this.embeddings,
      { pineconeIndex: index }
    );
  }

  async storeMemory(memory) {
    const { content, userId, nickname, type } = memory;
    await this.vectorStore.addDocuments([{
      pageContent: content,
      metadata: {
        userId,
        nickname,
        type,
        timestamp: new Date().toISOString()
      }
    }]);
  }

  async searchMemories(query, filter = {}) {
    const results = await this.vectorStore.similaritySearch(query, 5, filter);
    return results;
  }

  async getUserMemories(userId) {
    return await this.searchMemories("", { userId });
  }

  async getContextualMemories(content) {
    return await this.searchMemories(content);
  }
}

module.exports = SamaraMemory; 