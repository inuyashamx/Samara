const { Client, GatewayIntentBits, Events } = require('discord.js');
require('dotenv').config();

// Check if Discord token is set
if (!process.env.DISCORD_TOKEN) {
  console.error('ERROR: DISCORD_TOKEN is not set in your .env file');
  process.exit(1);
}

// Create a simple Discord client
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.GuildMembers,
  ],
});

// Log when the bot is ready
client.once(Events.ClientReady, (readyClient) => {
  console.log(`✅ Discord bot successfully connected as ${readyClient.user.tag}`);
  console.log(`Bot ID: ${readyClient.user.id}`);
  console.log(`Invite URL: https://discord.com/oauth2/authorize?client_id=${readyClient.user.id}&scope=bot&permissions=274878221376`);
  
  // List all servers the bot is in
  console.log('\nConnected to the following servers:');
  readyClient.guilds.cache.forEach(guild => {
    console.log(`- ${guild.name} (ID: ${guild.id})`);
  });
  
  console.log('\nTest complete! The bot is properly connected to Discord.');
  console.log('Press Ctrl+C to exit.');
});

// Simple message handler to test responses
client.on(Events.MessageCreate, (message) => {
  // Ignore messages from bots to prevent loops
  if (message.author.bot) return;
  
  // Log all messages received (for diagnostic purposes)
  console.log(`Message received from ${message.author.tag}: ${message.content}`);
  
  // Check if the message mentions the bot or contains its name
  const mentionsSamara = message.mentions.users.has(client.user.id) || 
    message.content.toLowerCase().includes('samara');
  
  if (mentionsSamara) {
    console.log('Bot was mentioned! Sending test response...');
    message.reply('I hear you! This is a test response from Samara.');
  }
});

// Log any errors
client.on(Events.Error, (error) => {
  console.error('Discord client error:', error);
});

// Connect to Discord
console.log('Connecting to Discord...');
client.login(process.env.DISCORD_TOKEN)
  .catch(error => {
    console.error('Failed to connect to Discord:', error);
  });
