#!/bin/bash
# Example shell script for using this service.
# Replace with actual code examples for your service.

curl -X POST https://api.example.com/v1/service \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "example input data",
    "parameters": {
      "option1": "value1",
      "option2": "value2"
    }
  }'
