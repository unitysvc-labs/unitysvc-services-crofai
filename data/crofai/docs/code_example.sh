curl {{ endpoint }}/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {{ api_key }}" \
  -d '{
    "model": "{{ model_name }}",
    "messages": [
      {
        "role": "user",
        "content": "Say this is a test"
      }
    ]
  }'