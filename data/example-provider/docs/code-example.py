"""
Example Python code for using this service.
Replace with actual code examples for your service.
"""

import requests

# Example API call
response = requests.post(
    "https://api.example.com/v1/service",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "input": "example input data",
        "parameters": {
            "option1": "value1",
            "option2": "value2"
        }
    }
)

result = response.json()
print(result)
