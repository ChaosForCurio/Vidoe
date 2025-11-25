const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

const API_URL = 'http://localhost:8000';
const API_KEY = 'test-key';

async function testHealth() {
  try {
    const response = await axios.get(`${API_URL}/health`);
    console.log('Health Check:', response.data);
  } catch (error) {
    console.error('Health Check Failed:', error.message);
  }
}

async function testTxt2Vid() {
  console.log('\nTesting /txt2vid...');
  const form = new FormData();
  form.append('prompt', 'A robot painting a canvas');
  form.append('num_frames', 16);

  try {
    const response = await axios.post(`${API_URL}/txt2vid`, form, {
      headers: {
        'X-API-Key': API_KEY,
        ...form.getHeaders(),
      },
      responseType: 'stream',
    });

    const writer = fs.createWriteStream('output_txt2vid_js.mp4');
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', () => {
        console.log('Success! Saved to output_txt2vid_js.mp4');
        resolve();
      });
      writer.on('error', reject);
    });
  } catch (error) {
    console.error('Txt2Vid Failed:', error.message);
  }
}

async function main() {
  await testHealth();
  // await testTxt2Vid(); // Uncomment to run
}

main();
