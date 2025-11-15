const WebSocket = require('ws');
const fs = require('fs');

const sampleData = JSON.parse(fs.readFileSync('sample-data.json', 'utf8'));
let currentIndex = 0;

const wss = new WebSocket.Server({ port: 8080 });
console.log('run in 8080');

wss.on('connection', ws => {
  console.log('client connected');

  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(sampleData[currentIndex]));
      currentIndex = (currentIndex + 1) % sampleData.length;
    } else {
      clearInterval(interval);
    }
  }, 500);

  ws.on('close', () => {
    console.log('bye');
    clearInterval(interval);
  });
});