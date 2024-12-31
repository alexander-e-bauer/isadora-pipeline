import config
import pandas as pd
from flask import jsonify

from xyz.modules.llm.embedding_tools import embedding_model, embedding_generator, embedding_search
from xyz.modules.llm.embedding_tools.embedding_search import google

logger = config.logger
OAI = config.OAI
search_df = pd.DataFrame()

def read_code(update=False):
    if update:
        embedding_generator.create_embeddings_of_self()

    df = embedding_model.read_embedding(embedding_path='xyz/modules/llm/embedding_tools/embeddings/code_metadata.csv')
    return df


def read_directory(directory, name, update=False):
    if update:
        embedding_generator.create_embeddings_of_text(directory, name)

    df = embedding_model.read_embedding(embedding_path=directory)
    return df, name



def get_completion(conversation_history, user_input: str, conversation_id: str,
                                    system_input: str = 'You are a helpful assistant.',
                                    model: str = "gpt-4o", streaming: bool = False,
                                    print_message: bool = False) -> str:

    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    # Append user message to conversation history
    conversation_history[conversation_id].append({"role": "user", "content": user_input})

    messages = [
                   {"role": "system", "content": system_input},
               ] + conversation_history[conversation_id]


    try:
        completion = OAI.client.chat.completions.create(
            model=model,
            messages=messages
        )
        output = completion.choices[0].message.content

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        print(f"\nCompletion: \nPrompt: {user_input}\nResult: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def chat_completion_with_embeddings(conversation_history, user_input: str, df: pd.DataFrame, conversation_id: str,
                                    system_input: str = 'You are a helpful assistant.',
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
        output = OAI.client.chat.completions.create(
        model='gpt-4o',
        messages=messages
    )

        output = output.choices[0].message.content

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def organize(message, conversation_id, conversation_history, persona, df: pd.DataFrame = None):
    try:
        completion = get_completion(
            user_input=message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            print_message=True,
            system_input=persona)
        response = f"{completion}"
        logger.debug(f"Sending response: {response}")
        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500


def file_embeddings(message, conversation_id, conversation_history, persona, df: pd.DataFrame = None):
    try:
        completion = chat_completion_with_embeddings(user_input=message,
                                                     conversation_id=conversation_id,
                                                     df=df,
                                                     conversation_history=conversation_history,
                                                     print_message=True,
                                                     system_input=persona)
        response = f"{completion}"
        logger.debug(f"Sending response: {response}")
        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500


def search_embeddings(message, conversation_id, conversation_history, persona, df: pd.DataFrame = None):
    try:
        completion = chat_completion_with_embeddings(user_input=message,
                                                     conversation_id=conversation_id,
                                                     df=df,
                                                     conversation_history=conversation_history,
                                                     print_message=True,
                                                     system_input=persona)
        response = f"{completion}"
        logger.debug(f"Sending response: {response}")
        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

def jsonify_chat(data, conversation_history, df: pd.DataFrame = None):
    global search_df
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    function = data.get('function', '')
    if function == "embedding":
        persona = "You are a helpful assistant."
        tool = "embedding"
    elif function == "search":
        persona = "You are a helpful assistant."
        tool = "search"
    elif function == "pirate":
        persona = "You are a helpful assistant who can only respond with the vernacular of a swashbuckler."
        tool = None
    elif function == "shakespeare":
        persona = "You are a helpful assistant who can only respond with the vernacular of Shakespeare."
        tool = None
    else:
        persona = "You are a helpful assistant."
        tool = None

    logger.debug(f"Received chat request. \nMessage: {message}, \nConversation ID: {conversation_id},"
                 f"\nFunction: {function} \nPersona: {persona} \nTool: {tool}")
    logger.debug(f"Current conversation history: {conversation_history.get(conversation_id, [])}")

    if tool == "embedding":
        return file_embeddings(
            message=message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            persona=persona,
            df=df)
    elif tool == "search":
        # Check if there's existing conversation history
        if conversation_id not in conversation_history or not conversation_history[conversation_id]:
            # No existing history - perform Google search and create embeddings
            return search_embeddings(
                message=message,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                persona=persona,
                df=search_df
            )
        else:
            # Existing history - just the conversation history without new search
            return search_embeddings(
                message=message,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                persona=persona,
                df=search_df
            )
    elif tool is None:
        return organize(
            message=message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            persona=persona)
    else:
        return "\nError XI: embedding_tool.py --- jsonify_chat tool sorting error\n"


