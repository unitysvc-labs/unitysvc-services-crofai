/**
 * Example JavaScript code for using this service.
 * Replace with actual code examples for your service.
 */

async function callService() {
  const response = await fetch('https://api.example.com/v1/service', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer YOUR_API_KEY',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: 'example input data',
      parameters: {
        option1: 'value1',
        option2: 'value2'
      }
    })
  });

  const result = await response.json();
  console.log(result);
}

callService();
