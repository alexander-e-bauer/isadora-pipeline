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





def get_persona(function):
    if function == "none":
        return "You are a helpful assistant.", "none"
    elif function == "embedding":
        return "You are a helpful assistant that uses embeddings to improve the accuracy of your responses.", "embedding"
    elif function == "search":
        return "You are a helpful assistant that searches the internet to improve the accuracy of your responses.", "google-search"
    elif function == "completion":
        return "You are a helpful assistant that completes code to improve the accuracy of your responses.", "none"
    elif function == "pirate":
        return "You are a swashbuckling pirate than can only respond with the demeanor of such.", "none"
    elif function == "shakespeare":
        return "You are shakespeare the playwright and can only respond with the demeanor of such.", "none"
    else:
        return "You are a helpful assistant.", "none"



def get_completion(prompt, persona="You are a helpful assistant.", model="gpt-4o"):
    completion = OAI.client.chat.completions.create(
        model=model,
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
        output = get_completion(messages, persona=system_input, model=model)

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


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


def jsonify_chat(data, conversation_history, df: pd.DataFrame = None):
    global search_df
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    function = data.get('function', '')
    persona, tool = get_persona(function)

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
    elif tool == "google-search":
        # Check if there's existing conversation history
        if conversation_id not in conversation_history or not conversation_history[conversation_id]:
            # No existing history - perform Google search and create embeddings
            try:
                search_df = google(message, number=5, conversation_id=conversation_id)
                return embedding_search.chat_completion_with_embeddings(user_input=message,
                    conversation_id=conversation_id,
                    conversation_history=conversation_history,
                    system_input=persona,
                    df=search_df)
            except Exception as e:
                logger.error(f"Error in Google search: {str(e)}", exc_info=True)
                return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500
        else:
            # Existing history - just the conversation history without new search
            return embedding_search.chat_completion_with_embeddings(user_input=message,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                system_input=persona,
                df=search_df)
    else:
        return get_completion(message, conversation_id, persona)


