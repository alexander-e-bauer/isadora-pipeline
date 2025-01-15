import config
import pandas as pd
from flask import jsonify

from xyz.modules.llm.embedding_tools import embedding_model, embedding_generator, embedding_search
from xyz.modules.llm.embedding_tools.embedding_search import google
from xyz.modules.llm.browser_service import BrowserService

logger = config.logger
OAI = config.OAI
search_df = pd.DataFrame()

def get_browser_service():
    return BrowserService.get_instance()


def initialize_code_embeddings(update=False):
    """Load or update code embeddings from the codebase"""
    if update:
        embedding_generator.create_embeddings_of_self()

    df = embedding_model.read_embedding(embedding_path='xyz/modules/llm/embedding_tools/embeddings/code_metadata.csv')
    return df


def initialize_directory_embeddings(directory, name, update=False):
    """Load or update embeddings from a specific directory"""
    if update:
        embedding_generator.create_embeddings_of_text(directory, name)

    df = embedding_model.read_embedding(embedding_path=directory)
    return df, name


def process_basic_chat(conversation_history, user_input: str, conversation_id: str,
                       system_input: str = 'You are a helpful assistant.',
                       model: str = "gpt-4o", streaming: bool = False,
                       print_message: bool = False) -> str:
    """Process a basic chat interaction without embeddings"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

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

        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        print(f"\nCompletion: \nPrompt: {user_input}\nResult: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def process_embedding_enhanced_chat(conversation_history, user_input: str, df: pd.DataFrame, conversation_id: str,
                                    system_input: str = 'You are a helpful assistant.',
                                    model: str = "gpt-4o", streaming: bool = False,
                                    print_message: bool = False) -> str:
    """Process chat with context from embeddings"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    query_msg = embedding_model.query_message(user_input, df, model=model)

    if print_message:
        print(f"Query message: {query_msg}")

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
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        logger.debug(f"Updated conversation history: {conversation_history[conversation_id]}")
        return output
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def handle_basic_chat_response(message, conversation_id, conversation_history, persona, df: pd.DataFrame = None):
    """Handle basic chat interactions and format response"""
    try:
        completion = process_basic_chat(
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


def handle_embedding_chat_response(message, conversation_id, conversation_history, persona, df: pd.DataFrame = None):
    """Handle embedding-enhanced chat interactions and format response"""
    try:
        completion = process_embedding_enhanced_chat(
            user_input=message,
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


def process_chat_request(data, conversation_history, df: pd.DataFrame = None, browser_service=None):
    message = data.get('message', '')
    function = data.get('function', '')
    conversation_id = data.get('conversationId', 'default')
    window_mode = data.get('windowMode', 'browser')
    window_content = data.get('windowContent', '')

    logger.info(f"3: Parsed input data:\n"
                f"message: {message}\n"
                f"function: {function}\n"
                f"conversation_id: {conversation_id}\n"
                f"window_mode: {window_mode}\n"
                f"window_content: {window_content}\n\n")

    # Handle browser navigation requests based on message content
    if message.lower().startswith(('go to ', 'navigate to ', 'open ')):
        logger.info("3b: Browser navigation command detected in the message.")
        url = message.split(' ', 2)[-1].strip()
        logger.info(f"Url Identified: {url} \n")

        try:
            logger.debug(f"4b: Attempting to navigate to URL: {url}")
            result = browser_service.navigate_to_url(url)
            logger.info(f"6b: Successfully navigated to {url}.")
            return jsonify({
                "response": f"Navigated to {url}",
                "window_content": result,
                "window_mode": "browser"
            })
        except Exception as e:
            logger.error(f"Error during browser navigation: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error: {str(e)}"}), 500

    # Define persona based on the provided function
    persona_map = {
        "embedding": "You are a helpful assistant.",
        "mysterious arcane orb": "You are a mysterious arcane orb and can only respond as such.",
        "pirate": "You are a helpful assistant who can only respond with the vernacular of a swashbuckler.",
        "shakespeare": "You are a helpful assistant who can only respond with the vernacular of Shakespeare.",
    }

    persona = persona_map.get(function, "You are a helpful assistant.")
    tool = "embedding" if function == "embedding" else None

    logger.info("Configured persona and tool for the chat response.")
    logger.debug(f"Persona: {persona}, Tool: {tool}")
    logger.debug(
        f"Current conversation history for ID {conversation_id}: {conversation_history.get(conversation_id, [])}")

    # Handle embedding-based and basic chat responses
    if tool == "embedding":
        logger.info("Handling embedding chat response.")
        return handle_embedding_chat_response(
            message=message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            persona=persona,
            df=df
        )
    elif tool is None:
        logger.info("Handling basic chat response.")
        return handle_basic_chat_response(
            message=message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            persona=persona
        )
    else:
        logger.warning("No suitable tool found for the chat request. Returning None.")
        return None

