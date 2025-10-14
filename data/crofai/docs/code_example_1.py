import openai
import json

client = openai.OpenAI(base_url="{{ endpoint }}", api_key="{{ api_key }}")


# Example function
def echo_message(message):
    return f"Echo: {message}"


response = client.chat.completions.create(
    model="{{ model_name }}",
    messages=[{"role": "user", "content": "Repeat: Hello world"}],
    functions=[
        {
            "name": "echo_message",
            "description": "Echoes back the given message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to echo"}
                },
                "required": ["message"],
            },
        }
    ],
    function_call="auto",
)

if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    function_args = json.loads(response.choices[0].message.function_call.arguments)

    if function_name == "echo_message":
        result = echo_message(function_args["message"])
        print(result)
