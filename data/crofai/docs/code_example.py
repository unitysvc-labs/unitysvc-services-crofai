import openai

client = openai.OpenAI(
    base_url="{{ endpoint }}",
    api_key="{{ api_key }}",
)
response = client.chat.completions.create(
    model="{{ model_name }}",
    # Change the model name to the deepseek one
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
)
print(response.choices[0].message.content)
