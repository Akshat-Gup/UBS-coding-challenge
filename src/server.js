const express = require('express');
const cors = require('cors');

const { assignConcerts } = require('./service/assignment');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/', (req, res) => {
  res.status(200).json({ status: 'ok', message: 'UBS challenge server running' });
});

app.post('/assign', (req, res) => {
  try {
    const input = req.body || {};
    const result = assignConcerts(input);
    return res.status(200).json(result);
  } catch (error) {
    return res.status(400).json({ error: error.message || 'Invalid input' });
  }
});

app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Server listening on port ${PORT}`);
});


