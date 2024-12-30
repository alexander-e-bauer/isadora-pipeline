import config
import pandas as pd
from flask import jsonify

from xyz.modules.llm.embedding_tools import embedding_model, embedding_generator

logger = config.logger
OAI = config.OAI

def read_code(update=False):
    if update:
        embedding_generator.create_embeddings_of_self()

    df = embedding_model.read_embedding()
    return df


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


def chat_completion_with_embeddings(conversation_history, user_input: str, df: pd.DataFrame, conversation_id: str,
                                    system_input: str,
                                    model: str = "gpt-4o", streaming: bool = False,
                                    print_message: bool = False) -> str:

    """
    Performs chat completion using GPT, incorporating conversation history and document embeddings.
    """
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    # Create the query message using the dataframe
    query_msg = embedding_model.query_message(user_input, df, model=model)

    if print_message:
        print(f"Query message: {query_msg}")

    # Append user message to conversation history
    conversation_history[conversation_id].append({"role": "user", "content": query_msg})

    messages = [
                   {"role": "system", "content": system_input},
               ] + conversation_history[conversation_id]

    logger.debug(f"Messages sent to API: {messages}")

    try:
        output = get_completion(messages)

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def jsonify_chat(data, conversation_history, df: pd.DataFrame = None):
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    logger.debug(f"Received chat request. Message: {message}, Conversation ID: {conversation_id}")
    logger.debug(f"Current conversation history: {conversation_history.get(conversation_id, [])}")
    try:
        completion = chat_completion_with_embeddings(user_input=message,
                                                     conversation_id=conversation_id,
                                                     df=df,
                                                     conversation_history=conversation_history)
        response = f"{completion}"
        logger.debug(f"Sending response: {response}")
        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500


