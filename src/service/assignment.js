function validateInput(payload) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Payload must be an object');
  }
  const { customers, concerts, priority } = payload;
  if (!Array.isArray(customers) || !Array.isArray(concerts)) {
    throw new Error('customers and concerts must be arrays');
  }
  if (customers.length < 1 || concerts.length < 1) {
    throw new Error('customers and concerts must be non-empty');
  }
  if (priority && typeof priority !== 'object') {
    throw new Error('priority must be an object');
  }
}

function squaredDistance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function assignConcerts(payload) {
  validateInput(payload);
  const { customers, concerts, priority = {} } = payload;

  // Map of concert name to its booking center location
  const concertMap = new Map(concerts.map((c) => [c.name, c.booking_center_location]));

  const result = {};

  for (const customer of customers) {
    const customerName = customer.name;
    const creditCard = customer.credit_card;
    const preferredConcert = priority[creditCard];

    // If this customer's card has a priority mapping, honor it if the concert exists
    if (preferredConcert && concertMap.has(preferredConcert)) {
      result[customerName] = preferredConcert;
      continue;
    }

    // Otherwise assign the nearest concert by Euclidean distance (squared)
    let chosenConcert = null;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const concert of concerts) {
      const dist = squaredDistance(customer.location, concert.booking_center_location);
      if (dist < bestDist) {
        bestDist = dist;
        chosenConcert = concert.name;
      }
    }
    result[customerName] = chosenConcert;
  }

  return result;
}

module.exports = { assignConcerts };


