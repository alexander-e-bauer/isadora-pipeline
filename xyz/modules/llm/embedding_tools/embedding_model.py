import re
import pandas as pd
import numpy as np
from scipy import spatial
import ast
import tiktoken
from typing import List, Tuple, Optional, Union
import config

log = config.log
logger = config.logger
OAI = config.OAI

embedding_model = OAI.embedding3

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_embedding(embedding_path: str = "xyz/modules/llm/embedding_tools/embeddings/code_metadata.csv") -> pd.DataFrame:
    """
    Read embeddings from a CSV file.

    Args:
        embedding_path: Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame containing embeddings
    """
    try:
        df = pd.read_csv(
            embedding_path,
            index_col=0,
            converters={
                'embedding': lambda x: ast.literal_eval(x)
            }
        )
        logger.debug(f"Successfully read embeddings from {embedding_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading embeddings: {str(e)}", exc_info=True)
        raise


def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> Tuple[List[str], List[float]]:
    """
    Returns a list of strings and relatednesses, sorted from most related to least.

    Args:
        query: Query string to compare against
        df: DataFrame containing embeddings
        relatedness_fn: Function to compute relatedness
        top_n: Number of top results to return

    Returns:
        Tuple[List[str], List[float]]: Lists of strings and their relatedness scores
    """
    try:
        query_embedding_response = OAI.client.embeddings.create(
            model=embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding

        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses) if strings_and_relatednesses else ([], [])
        return list(strings[:top_n]), list(relatednesses[:top_n])
    except Exception as e:
        logger.error(f"Error in strings_ranked_by_relatedness: {str(e)}", exc_info=True)
        raise


def remove_stuff(text: str) -> str:
    """
    Clean text by removing unnecessary characters while preserving URLs.

    Args:
        text: Text to clean

    Returns:
        str: Cleaned text
    """
    try:
        # Pattern to identify URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)

        # Replace URLs with a placeholder
        placeholder = "URL_PLACEHOLDER"
        for url in urls:
            text = text.replace(url, placeholder)

        # Remove large spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Restore URLs
        for url in urls:
            text = text.replace(placeholder, url, 1)

        return text
    except Exception as e:
        logger.error(f"Error in remove_stuff: {str(e)}", exc_info=True)
        return text


def get_embedding(text_to_embed: str) -> List[float]:
    """
    Get embedding for a text string.

    Args:
        text_to_embed: Text to get embedding for

    Returns:
        List[float]: Embedding vector
    """
    try:
        text_to_embed = remove_stuff(text_to_embed)
        response = OAI.client.embeddings.create(
            model=embedding_model,
            input=[text_to_embed]
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text: {text_to_embed[:100]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}", exc_info=True)
        raise


def create_embedding_df(excel_path: str, embedding_path: str) -> None:
    """
    Create embeddings DataFrame from Excel file and save to CSV.

    Args:
        excel_path: Path to input Excel file
        embedding_path: Path to save embeddings CSV
    """
    try:
        df = pd.DataFrame()
        review_df = pd.read_excel(excel_path, header=None)

        df['text'] = review_df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df.reset_index(drop=True, inplace=True)

        df["embedding"] = df["text"].astype(str).apply(get_embedding)

        df.to_csv(embedding_path, index=False)
        logger.info(f"Successfully created embeddings and saved to {embedding_path}")
    except Exception as e:
        logger.error(f"Error creating embedding DataFrame: {str(e)}", exc_info=True)
        raise


def num_tokens(text: str, model: str = OAI.gpt4o) -> int:
    """
    Count the number of tokens in a string.

    Args:
        text: Text to count tokens for
        model: Model to use for tokenization

    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}", exc_info=True)
        return 0


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str = OAI.gpt4o,
        token_budget: int = 3000
) -> str:
    """
    Create a message for GPT with relevant context from the DataFrame.

    Args:
        query: User query
        df: DataFrame containing embeddings
        model: Model to use
        token_budget: Maximum tokens to use

    Returns:
        str: Formatted message with context
    """
    try:
        strings, relatednesses = strings_ranked_by_relatedness(query, df)

        introduction = 'Use the Documents provided below to answer the questions. '
        question = f"\n\nTask: {query}"
        message = introduction

        for string in strings:
            next_article = f'\n\nOriginal Code File:\n"""\n{string}\n"""'
            if num_tokens(message + next_article + question, model=model) > token_budget:
                break
            message += next_article

        return message + question
    except Exception as e:
        logger.error(f"Error creating query message: {str(e)}", exc_info=True)
        raise


def query_message_code(
        query: str,
        df: pd.DataFrame,
        model: str = OAI.gpt4o,
        token_budget: int = 3000
) -> str:
    """
    Create a code-specific message for GPT with relevant context.

    Args:
        query: User query
        df: DataFrame containing embeddings
        model: Model to use
        token_budget: Maximum tokens to use

    Returns:
        str: Formatted message with context
    """
    try:
        if df.empty:
            return "DataFrame is empty. Cannot generate message."

        strings, relatednesses = strings_ranked_by_relatedness(query, df)

        introduction = (
            'Use the Original Code Files provided below to answer the questions about the code. '
            'Based on the input, Generate one single code that implements an improvement '
            'upon the original code. Ensure compatibility with the original code. '
            'Respond only with the code, no additional information.'
        )

        question = f"\n\nTask: {query}"
        message = introduction

        for string in strings:
            next_article = f'\n\nOriginal Code File:\n"""\n{string}\n"""'
            if num_tokens(message + next_article + question, model=model) > token_budget:
                break
            message += next_article

        return message + question
    except Exception as e:
        logger.error(f"Error creating code query message: {str(e)}", exc_info=True)
        raise


def ask(
        query: str,
        df: pd.DataFrame,
        conversation_id: str,
        model: str = OAI.gpt4o,
        print_message: bool = False,
) -> str:
    """
    Answers a query using GPT and a dataframe of relevant texts and embeddings.

    Args:
        query: User query
        df: DataFrame containing embeddings
        conversation_id: ID for the conversation
        model: Model to use
        print_message: Whether to print debug messages

    Returns:
        str: Response from the model
    """
    try:
        message = query_message(query, df, model=model)
        if print_message:
            logger.debug(f"Generated message: {message[:200]}...")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]

        response = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            conversation_id=conversation_id,
            temperature=1.9
        )

        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        logger.error(f"Error in ask function: {str(e)}", exc_info=True)
        raise


def ask_familiar(
        query: str,
        df: pd.DataFrame,
        conversation_id: str,
        model: str = OAI.gpt4o,
        print_message: bool = False,
) -> str:
    """
    Answers a query using GPT with a more familiar tone.

    Args:
        query: User query
        df: DataFrame containing embeddings
        conversation_id: ID for the conversation
        model: Model to use
        print_message: Whether to print debug messages

    Returns:
        str: Response from the model
    """
    try:
        message = query_message(query, df, model=model)
        if print_message:
            logger.debug(f"Generated message: {message[:200]}...")

        system_content = (
            "You are a helpful assistant who is trying to get the person whose resume and work "
            "is represented in the provided documents a job as a data scientist or web developer. "
            "You respond in a professional, witty, and honest manner and "
            "provide specific examples whenever possible. Speak in a general manner, as you "
            "are open to many opportunities, not just one specific position, and could be "
            "contacted by anybody including potential employers. Don't respond with too many "
            "words, don't use verbose grammar, and don't oversell yourself. Don't give out "
            "too much information without being prompted to do so."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message},
        ]

        response = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            conversation_id=conversation_id,
            temperature=0
        )

        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        logger.error(f"Error in ask_familiar function: {str(e)}", exc_info=True)
        raise


def ask_code(
        query: str,
        df: pd.DataFrame,
        model: str = OAI.gpt4o,
        print_message: bool = False,
) -> str:
    """
    Answers a code-related query using GPT.

    Args:
        query: User query
        df: DataFrame containing embeddings
        model: Model to use
        print_message: Whether to print debug messages

    Returns:
        str: Response from the model
    """
    try:
        message = query_message_code(query, df, model=model)
        if print_message:
            logger.debug(f"Generated message: {message[:200]}...")

        messages = [
            {
                "role": "system",
                "content": "You Complete the users python, javascript, css, and html code "
                           "and fully Implement New Code if possible"
            },
            {"role": "user", "content": message},
        ]

        response = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        logger.error(f"Error in ask_code function: {str(e)}", exc_info=True)
        raise
