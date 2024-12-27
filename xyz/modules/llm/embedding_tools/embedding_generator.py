import os
from pathlib import Path
import docx
from PyPDF2 import PdfReader
import re
import pandas as pd

import markdown
from bs4 import BeautifulSoup
import tiktoken
import openai
import config
from xyz.modules.llm.embedding_tools.embedding_model import embedding_model, create_embedding_df, read_embedding

log = config.log
OAI = config.OAI


def num_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o')
    encoding = encoding.encode(text, disallowed_special=())
    print(len(encoding))
    return len(encoding)


def save_text_to_file(text, file_path):
    file_path = Path(file_path)
    with open(file_path, 'w') as file:
        file.write(text)



def remove_stuff(text: str) -> str:
    """Remove punctuation (except in URLs), newline, tab characters, and large spaces."""
    # Pattern to identify URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Find all URLs using the pattern
    urls = re.findall(url_pattern, text)
    # Replace URLs with a placeholder to avoid altering them
    placeholder = "URL_PLACEHOLDER"
    for url in urls:
        text = text.replace(url, placeholder)

    # Remove large spaces (5 or more spaces)
    text = re.sub(r' {5,}', ' ', text)

    # Restore URLs from placeholders
    for url in urls:
        text = text.replace(placeholder, url, 1)

    return text



def get_embedding(text_to_embed):
    text_to_embed = remove_stuff(text_to_embed)
    # Embed a line of text
    response = OAI.client.embeddings.create(
        model=embedding_model,
        input=[text_to_embed]
    )
    # Extract the AI output embedding as a list of floats
    embedding = response.data[0].embedding
    print(f"---\nEmbedding: {embedding} \nText: {text_to_embed}")

    return embedding


def get_source_code(directory):
    """Returns the source code of all Python files in a directory."""
    df = pd.DataFrame(columns=['filepath', 'text'])
    string = []
    paths = []
    for root, dirs, files in os.walk(directory):
        if root.startswith('./lib'):
            continue
        elif root.startswith('./bin'):
            continue
        elif root.startswith('./include'):
            continue
        else:
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    raw_text = read_file_as_raw_text(file_path)
                elif file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    raw_text = read_file_as_raw_text(file_path)
                elif file.endswith(".css"):
                    file_path = os.path.join(root, file)
                    raw_text = read_file_as_raw_text(file_path)
                elif file.endswith(".js"):
                    file_path = os.path.join(root, file)
                    raw_text = read_file_as_raw_text(file_path)

                else:
                    continue
                log(f"Source code: {file_path}")
                string.append(raw_text), paths.append(file_path)

                df = df._append({'filepath': file_path, 'text': raw_text}, ignore_index=True)

    return df




def get_document_text(directory):
    """
    Returns the text content and embeddings of all Word documents, PDFs, Markdown files,
    and HTML files in a directory, or from an Excel file if provided.
    """
    df = pd.DataFrame(columns=['filepath', 'text', 'embedding'])

    # Process documents in directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('~$'):
                continue

            if file.endswith(('.doc', '.docx', '.pdf', '.md', '.html')):
                file_path = os.path.join(root, file)

                try:
                    if file.endswith(('.doc', '.docx')):
                        raw_text = read_word_document(file_path)  # You need to implement this function
                    elif file.endswith('.pdf'):
                        raw_text = read_pdf_document(file_path)  # You need to implement this function
                    elif file.endswith('.md'):
                        raw_text = read_markdown_file(file_path)
                    elif file.endswith('.html'):
                        raw_text = read_html_file(file_path)

                    print(f"Processed document: {file_path}")
                    df = df._append({'filepath': file_path, 'text': raw_text}, ignore_index=True)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    if df.empty:
        print("No valid documents found.")
        return df

    # Generate embeddings using the get_embedding function
    df['embedding'] = df['text'].astype(str).apply(get_embedding)

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


def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

source_code = ''


def get_completion(text, script=source_code):
    completion = openai.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You take python, javascript, css, html, and sql code as input "
                                          "complete/improve/remove errors from the code, "
                                          "and return the finished code as output. If there is something that could "
                                          "be added to the file to make it better, "
                                          "attempt to integrate it into the code."
                                          f"The code in its entirety is given below:\n{script}"},
            {"role": "user", "content": f"{text}"}
        ]
    )

    result = completion.choices[0].message.content
    log(f"\nCompletion: \nPrompt: {text}\nResult: {result}")
    return result


def create_excel_file(filepath):
    # Save the DataFrame to an Excel file
    df = get_source_code(Path())

    df.to_excel(filepath, index=False, header=False, engine='openpyxl')
    return df


def create_excel_file_text(target_directory, filepath):
    # Save the DataFrame to an Excel file
    df = get_document_text(target_directory)

    df.to_excel(filepath, index=False, header=False, engine='openpyxl')
    return df


def save_source():
    global source_code
    filepath = Path('code/source_code.txt')
    source_code = get_source_code(Path())
    source_code = '\n'.join(source_code)
    num_tokens(source_code)
    save_text_to_file(source_code, filepath)

def create_embeddings_of_self():
    create_excel_file('./embeddings/code_metadata.xlsx')
    create_embedding_df('./embeddings/code_metadata.xlsx',
                                         './embeddings/code_metadata.csv')

def create_embeddings_of_text(target, name):
    create_excel_file_text(target, f'./embeddings/{name}.xlsx')
    create_embedding_df(f'./embeddings/{name}.xlsx',
                                         f'./embeddings/{name}.csv')


def chat_completion_with_embeddings(conversation_history, user_input: str, df: pd.DataFrame, conversation_id: str,
                                    system_input: str = "You are a data scientist named Alex Bauer who is presenting "
                                                        "his projects online in order to get a professional job.",
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

    log(f"Messages sent to API: {messages}")

    try:
        completion = OAI.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=streaming,
            temperature=0
        )

        if not streaming:
            output = completion.choices[0].message.content
        else:
            output = ""
            for chunk in completion:
                output += str(chunk.choices[0].delta.content or '')
                print(chunk.choices[0].delta.content or '', end='', flush=True)

        # Append assistant's response to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": output})

        log(f"Updated conversation history: {conversation_history[conversation_id]}")
        return output
    except Exception as e:
        log(f"Error in chat completion: {str(e)}")
        raise

#df = read_embedding('embeddings/resume_test.csv')

#print(df)
#answer = ask_familiar("explain this code", df=df, print_message=True, conversation_id='conversation-1727902220357-g6xwoelao')
#print(answer)


#create_embeddings_of_text('../llm/knowledge_sources/personal', 'resume_test')

def run(update=False):
    if update:
        create_embeddings_of_self()
    else:
        df = read_embedding('embeddings/code_metadata.csv')


def save_embeddings(directory, output_path):
    df = get_document_text(directory)  # For processing documents in a directory
    print(df.head())
    df.to_csv(output_path, index=False)

run(True)

#directory = "knowledge_sources/personal"
#output_path = "embeddings/resume_test.csv"
#save_embeddings(directory, output_path)

#df = read_embedding('embeddings/resume_test.csv')
#rint(df)
#answer = ask("talk to me about this resume", df=df, print_message=True, conversation_id='conversation-1727902220357-g6xwoelao')
#print(answer)
