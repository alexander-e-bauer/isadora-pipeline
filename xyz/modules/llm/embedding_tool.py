import config
import pandas as pd
from flask import jsonify
from typing import Dict, Any, Optional, List
from xyz.modules.llm.embedding_tools import embedding_model, embedding_generator

logger = config.logger
OAI = config.OAI


def read_code(update: bool = False) -> pd.DataFrame:
    """
    Read code embeddings from file, optionally updating them first.

    Args:
        update (bool): Whether to regenerate embeddings before reading

    Returns:
        pd.DataFrame: DataFrame containing code embeddings
    """
    if update:
        embedding_generator.create_embeddings_of_self()

    df = embedding_model.read_embedding('embeddings/code_metadata.csv')
    return df


def get_completion(messages: List[Dict[str, str]], model: str = 'gpt-4o') -> str:
    """
    Get completion from OpenAI API.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries
        model (str): Model to use for completion

    Returns:
        str: Completion result
    """
    try:
        completion = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = completion.choices[0].message.content
        logger.debug(f"Completion result for model {model}: {result[:100]}...")
        return result

    except Exception as e:
        logger.error(f"Error in get_completion: {str(e)}", exc_info=True)
        raise


def chat_completion_with_embeddings(
        conversation_history: Dict[str, List[Dict[str, str]]],
        user_input: str,
        df: pd.DataFrame,
        conversation_id: str,
        system_input: str = "You are a data scientist named Alex Bauer who is presenting "
                            "his projects online in order to get a professional job.",
        model: str = "gpt-4o",
        streaming: bool = False,
        print_message: bool = False,
        max_history: int = 10
) -> str:
    """
    Performs chat completion using GPT, incorporating conversation history and document embeddings.

    Args:
        conversation_history: Dictionary storing conversation history
        user_input: User's input message
        df: DataFrame containing embeddings
        conversation_id: Unique identifier for the conversation
        system_input: System prompt
        model: Model to use for completion
        streaming: Whether to use streaming response
        print_message: Whether to print debug messages
        max_history: Maximum number of messages to keep in history

    Returns:
        str: Assistant's response
    """
    try:
        # Initialize conversation history if it doesn't exist
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []

        # Trim conversation history if it exceeds max_history
        if len(conversation_history[
                   conversation_id]) > max_history * 2:  # *2 because each exchange has user + assistant message
            conversation_history[conversation_id] = conversation_history[conversation_id][-max_history * 2:]

        # Create the query message using the dataframe
        query_msg = embedding_model.query_message(user_input, df, model=model)

        if print_message:
            logger.debug(f"Query message: {query_msg}")

        # Append user message to conversation history
        conversation_history[conversation_id].append({
            "role": "user",
            "content": query_msg
        })

        # Prepare messages array
        messages = [
                       {"role": "system", "content": system_input},
                   ] + conversation_history[conversation_id]

        logger.debug(f"Sending {len(messages)} messages to API")

        # Get completion
        output = get_completion(messages, model)

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({
            "role": "assistant",
            "content": output
        })

        logger.debug(f"Updated conversation history length: {len(conversation_history[conversation_id])}")
        return output

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise


def jsonify_chat(
        data: Dict[str, Any],
        conversation_history: Dict[str, List[Dict[str, str]]],
        df: Optional[pd.DataFrame] = None
) -> tuple:
    """
    Handle chat requests and return JSON response.

    Args:
        data: Request data containing message and conversation ID
        conversation_history: Dictionary storing conversation history
        df: DataFrame containing embeddings

    Returns:
        tuple: JSON response and status code
    """
    try:
        message = data.get('message', '')
        conversation_id = data.get('conversationId', 'default')
        model = data.get('model', 'gpt-4o')

        if not message:
            return jsonify({"error": "No message provided"}), 400

        if df is None:
            df = read_code()

        logger.debug(f"Processing chat request - Message: {message[:100]}... ID: {conversation_id}")
        logger.debug(f"Current conversation length: {len(conversation_history.get(conversation_id, []))}")

        completion = chat_completion_with_embeddings(
            user_input=message,
            conversation_id=conversation_id,
            df=df,
            conversation_history=conversation_history,
            model=model
        )

        response = {
            "response": completion,
            "conversation_id": conversation_id,
            "status": "success"
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in jsonify_chat: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e),
            "status": "error"
        }), 500


def clear_conversation(conversation_id: str, conversation_history: Dict[str, List[Dict[str, str]]]) -> tuple:
    """
    Clear the conversation history for a given conversation ID.

    Args:
        conversation_id: ID of conversation to clear
        conversation_history: Dictionary storing conversation history

    Returns:
        tuple: JSON response and status code
    """
    try:
        if conversation_id in conversation_history:
            conversation_history[conversation_id] = []
            return jsonify({"status": "success", "message": "Conversation cleared"}), 200
        return jsonify({"status": "error", "message": "Conversation ID not found"}), 404
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
