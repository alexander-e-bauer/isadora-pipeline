import requests


def send_chat_request(message, conversation_id='default'):
    # Define the URL with the specific endpoint
    url = 'http://localhost:5000/api/chat'

    # Define headers including Origin for CORS
    headers = {
        'Content-Type': 'application/json',
        'Origin': 'http://localhost'  # Adjust this based on your ALLOWED_ORIGINS
    }

    # Data structure matching jsonify_chat requirements
    data = {
        "message": message,
        "conversationId": conversation_id
    }

    try:
        # Send POST request
        response = requests.post(url, json=data, headers=headers)

        # Check if request was successful
        if response.status_code == 200:
            response_data = response.json()
            print('Chat request successful!')
            print('Response:', response_data.get('response'))
        elif response.status_code == 500:
            error_data = response.json()
            print(f'Server error: {error_data.get("error")}')
        else:
            print(f'Request failed with status code: {response.status_code}')
            print('Response:', response.text)

    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    # Example usage
    message = "Hello, how are you?"
    # You can optionally specify a conversation_id
    # send_chat_request(message, conversation_id="custom_conversation_1")
    send_chat_request(message)
