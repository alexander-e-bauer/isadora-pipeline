import config
import pandas as pd
import pprint
from flask import jsonify
import json

from xyz.modules.llm.embedding_tools import embedding_model, embedding_generator, embedding_search
from xyz.modules.llm.embedding_tools.embedding_search import google
from xyz.modules.llm.browser_service import BrowserService

logger = config.logger
OAI = config.OAI
search_df = pd.DataFrame()

def generate_dynamic_response(title: str, url: str, summary: str) -> str:
    """
    Use OpenAI's GPT model to generate a conversational response dynamically
    based on the retrieved webpage data.
    """
    if not title and not summary:
        return f"I navigated to {url}, but I couldn't find much information there."

    # Construct the prompt for GPT
    prompt = (
        f"I visited the website '{title}' at {url}. "
        f"Here's a summary of what I found: {summary}. "
        "Please generate a conversational response for the user, summarizing this information naturally."
    )

    try:
        # Call OpenAI's chat completion API
        completion = OAI.client.chat.completions.create(
            model="gpt-4o",  # Use the model defined in your setup
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates conversational responses."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and return the generated response
        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        logger.error(f"Error generating dynamic response with OpenAI: {str(e)}", exc_info=True)
        return "I encountered an error while generating a response. Please try again later."


browser_tool_schema = {
    "name": "navigate_to_url",
    "description": "Navigate to a URL using the browser.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "format": "uri",
                "pattern": "^https?://",
                "description": "The full URL to navigate to (must include http/https)"
            }
        },
        "required": ["url"]
    }
}


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
        # Include the function schema in the API call
        completion = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[browser_tool_schema],
            function_call="auto"
        )

        response_message = completion.choices[0].message

        # Check if the model wants to call a function
        if response_message.function_call:
            function_call = response_message.function_call
            logger.info(f"Function call detected: {function_call}")

            # Handle the function call
            if function_call.name == "navigate_to_url":
                function_args = json.loads(function_call.arguments)
                url = function_args.get("url")
                navigation_result = handle_browser_navigation(url)

                # Add function call response to conversation history
                conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": f"Function call: Navigated to {url}",
                    "function_call": {
                        "name": function_call.name,
                        "arguments": function_call.arguments
                    }
                })

                if navigation_result["status"] == "success":
                    page_data = navigation_result["data"]

                    # Dynamically craft a conversational response using OpenAI
                    response = generate_dynamic_response(
                        title=page_data.get("title"),
                        url=page_data.get("url"),
                        summary=page_data.get("summary")
                    )
                    return response
                else:
                    # Handle error case
                    return f"Sorry, I couldn't navigate to {url}. Here's the error: {navigation_result['message']}"

        # Handle normal response
        output = response_message.content
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        if print_message:
            print(f"\nCompletion: \nPrompt: {user_input}\nResult: {output}")

        return output

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise



def handle_browser_navigation(url: str):
    """Handle the browser navigation request."""
    try:
        logger.info(f"Attempting to navigate to URL: {url}")
        browser_service = get_browser_service()

        if not browser_service.check_status():
            logger.info("Browser is not open. Starting the browser...")
            browser_service.start_browser()

        result = browser_service.navigate_to_url(url)
        logger.info(f"Successfully navigated to {url}.")
        print(f"!!!!! {result}")
        return result

    except Exception as e:
        logger.error(f"Navigation error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "status_code": 500
        }


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
        # Include the function schema in the API call
        completion = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[browser_tool_schema],  # Add the browser tool schema
            function_call="auto"  # Let the model decide when to call the function
        )

        # Check if the model wants to call a function
        if completion.choices[0].finish_reason == "function_call":
            function_call = completion.choices[0].message.function_call
            logger.info(f"Function call detected: {function_call}")

            # Handle the function call
            if function_call["name"] == "navigate_to_url":
                function_args = json.loads(function_call["arguments"])
                url = function_args.get("url")
                return handle_browser_navigation(url)

        # Otherwise, return the model's response
        output = completion.choices[0].message.content
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        print(f"\nCompletion: \nPrompt: {user_input}\nResult: {output}")
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
    pprint.pprint(data)
    message = data.get('message', '')
    function = data.get('function', '')
    conversation_id = data.get('conversation_id', '')
    window_mode = data.get('window_mode', '')
    window_content = data.get('current_window_content', '')

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
        logger.info(f"URL Identified: {url} \n")

    # Define persona based on the provided function
    persona_map = {
        "default": "You are a helpful assistant.",
        "browser": "You are a helpful internet browsing assistant who controls a selenium browser operating on a VM.",
        "data": "You are a helpful data oracle, who can scrape the web for data and create reports and dashboards.",
        "security": "You are a helpful security assistant who has access to a database of bot fingerprint embeddings.",
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

