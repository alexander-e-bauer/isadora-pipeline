import config
from flask import Blueprint

OAI = config.OAI

ai = Blueprint('ai', __name__, template_folder='templates')

openai_client = OAI.client


def init_app(app):
    app.register_blueprint(ai, url_prefix='/llm')


def get_completion(prompt, persona="You are a helpful assistant."):
    completion = OAI.client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system",
             "content": f"{persona}"},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    result = completion.choices[0].message.content
    print(f"\nCompletion: \nPrompt: {prompt}\nResult: {result}")
    return result


def get_completion_(prompt, conversation_history, conversation_id: str, persona="You are a helpful assistant.",
                   model="gpt-4o"):
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    completion = OAI.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f"{persona}"},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    # Append user message to conversation history
    conversation_history[conversation_id].append({"role": "user", "content": prompt})

    messages = [
                   {"role": "system", "content": persona},
               ] + conversation_history[conversation_id]

    result = completion.choices[0].message.content
    print(f"\nCompletion: \nPrompt: {prompt}\nResult: {result}")
    return result


@ai.route('/chat', methods=['POST'])
def chat_completion(user_input, system_input="you are a helpful assistant", image_path=None, tools=None,
                    streaming=False):
    if image_path is None:
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=streaming
        )
    else:
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": [
                {"type": "image", "data": image_path},
                {"type": "text", "text": user_input}
            ]}
        ]
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=streaming
        )
    if not streaming:
        output = completion.choices[0].message.content
        print(output)
        return output
    else:
        output = ""
        for chunk in completion:
            output += str(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content)
        print(output)
        return output


@ai.route('/tts', methods=['POST'])
def text_to_speech(text, file_name="speech"):
    speech_file_path = f"static/audio/{file_name}.mp3"
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=f"{text}",
        speed=0.9
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path, response


@ai.route('/respond', methods=['POST'])
def respond(user_input):
    response = chat_completion(user_input)
    file_path, response = text_to_speech(response)
    return file_path, response


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
]




