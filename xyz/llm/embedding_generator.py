import os
from pathlib import Path
import docx
from PyPDF2 import PdfReader
import re
import pandas as pd
from scipy import spatial
import ast
import tiktoken
import markdown
from bs4 import BeautifulSoup
import tiktoken
import openai
import config
import xyz.llm.embedding_model as flask_embeddings
from xyz.llm.embedding_model import embedding_model, read_embedding, remove_stuff


log = config.log
OAI = config.OAI


def num_tokens(text, model="gpt-4"):
    """
    Count the number of tokens in a text string for a given model.

    Args:
        text (str): The text to count tokens for
        model (str): The model name (defaults to gpt-4)

    Returns:
        int: Number of tokens
    """
    try:
        # Map model names to supported encodings
        model_to_encoding = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "cl100k_base",
            "gpt-4o-mini": "cl100k_base",
            "gpt-4.1": "cl100k_base",
            "gpt-4.1-nano": "cl100k_base",
            "gpt-4-32k": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-3.5-turbo-16k": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-3-large": "cl100k_base",
        }

        # Get the encoding name, default to cl100k_base for unknown models
        encoding_name = model_to_encoding.get(model, "cl100k_base")

        # Get the encoding
        encoding = tiktoken.get_encoding(encoding_name)

        # Count tokens
        return len(encoding.encode(str(text)))

    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimate (1 token ≈ 4 characters for GPT models)
        return len(str(text)) // 4


def save_text_to_file(text, file_path):
    file_path = Path(file_path)
    with open(file_path, 'w') as file:
        file.write(text)


def get_embedding(text_to_embed):
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    text_to_embed = remove_stuff(text_to_embed)
    # Check the number of tokens
    token_count = num_tokens(text_to_embed)
    max_token_limit = 8192  # Adjust based on your model's token limit

    if token_count is None or token_count > max_token_limit:
        print(f"Text exceeds the token limit ({max_token_limit} tokens). Skipping embedding.")
        return None

    try:
        # Embed a line of text
        response = OAI.client.embeddings.create(
            model=embedding_model,
            input=[text_to_embed]
        )
        # Extract the AI output embedding as a list of floats
        embedding = response.data[0].embedding
        print(f"---\nEmbedding generated successfully for text: {text_to_embed[:100]}...")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def get_document_text(directory):
    """
    Returns the text content and embeddings of all Word documents, PDFs, Markdown files,
    and HTML files in a directory, or from an Excel file if provided.
    """
    df = pd.DataFrame(columns=['filepath', 'text', 'embedding'])

    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {os.path.abspath(directory)}")

    # Process documents in directory
    for root, dirs, files in os.walk(directory):
        print(f"Scanning directory: {root}")
        for file in files:
            if file.startswith('~$'):  # Skip temporary files
                continue

            if file.endswith(('.doc', '.docx', '.pdf', '.md', '.html')):
                file_path = os.path.join(root, file)

                try:
                    if file.endswith(('.doc', '.docx')):
                        raw_text = read_word_document(file_path)
                    elif file.endswith('.pdf'):
                        raw_text = read_pdf_document(file_path)
                    elif file.endswith('.md'):
                        raw_text = read_markdown_file(file_path)
                    elif file.endswith('.html'):
                        raw_text = read_html_file(file_path)
                    else:
                        print("No Readable Documents Found (.docx, .pdf, .md, .html)")
                        return

                    if raw_text.strip():  # Check if the extracted text is not empty
                        print(f"Processed document: {file_path}")
                        embedding = get_embedding(raw_text)
                        df = df._append({'filepath': file_path, 'text': raw_text, 'embedding': embedding}, ignore_index=True)
                    else:
                        print(f"No text found in document: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    if df.empty:
        print("No valid documents found.")
        return df

    return df



def read_markdown_file(file_path):
    """
    Read and convert a Markdown file to plain text.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def read_html_file(file_path):
    """
    Read an HTML file and extract its text content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()


def read_word_document(file_path):
    """Reads the text content of a Word document."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def read_pdf_document(file_path):
    """Reads the text content of a PDF document."""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        full_text = []
        for page in pdf_reader.pages:
            full_text.append(page.extract_text())
    return '\n'.join(full_text)


def read_file_as_raw_text(file_path):
    """Reads a file and returns its contents as a raw string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."


def create_excel_file_text(target_directory, filepath):
    # Save the DataFrame to an Excel file
    df = get_document_text(target_directory)

    df.to_excel(filepath, index=False, header=False, engine='openpyxl')
    return df


def create_embeddings_of_text(target, name):
    create_excel_file_text(target, f'../llm/embeddings/{name}.xlsx')
    flask_embeddings.create_embedding_df(f'../llm/embeddings/{name}.xlsx',
                                         f'../llm/embeddings/{name}.csv')


def relatedness(prompt):
    rel = flask_embeddings.relatedness_score(prompt)
    print(rel)
    return rel


def save_embeddings(directory, output_path):
    """
    Processes documents in a directory, generates embeddings, and saves them to a CSV file.
    """
    # Ensure the parent directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Process and save embeddings
    df = get_document_text(directory)  # For processing documents in a directory
    if df is None or df.empty:
        raise ValueError("No valid documents found in the directory.")

    print("Sample of processed data:")
    print(df.head())

    # Save DataFrame to CSV
    df.to_csv(output_path, index=False)





#directory = "xyz/llm/knowledge_sources/personal"
#output_path = "xyz/llm/embeddings/resume_test.csv"

#save_embeddings(directory, output_path)

#df = read_embedding('xyz/llm/embeddings/resume_test.csv')

def main():
    """
    Main function to test the num_tokens function.
    """
    # Load the encoding for the GPT-4 model
    model_name = "gpt-4"
    encoding = tiktoken.encoding_for_model(model_name)

    # Define test cases
    test_cases = [
        "The quick brown fox jumps over the lazy dog",  # Normal sentence
        "Hello, World!",  # A greeting with punctuation
        "This is a test case with 8 words in total",  # Count tokens
        "",  # Empty input
        "SingleWord",  # A single token
        "Multiple    spaces    between words",  # Sentence with extra spaces
        " punctuations!shouldn't. affect: counting,tokens ",  # Sentence with punctuation
    ]

    print(f"Running tests for num_tokens using encoding for model: {model_name}")
    for i, input_text in enumerate(test_cases, 1):
        try:
            result = num_tokens(input_text, encoding)
            print(f"Test #{i}: PASSED | Input: '{input_text}' | Tokens: {result}")
        except Exception as e:
            print(f"Test #{i}: FAILED | Input: '{input_text}' | Unexpected error: {e}")

    print("Testing completed.")

if __name__ == "__main__":
    main()