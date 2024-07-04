from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import os
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

chat_sessions = {}  # Dictionary to store chat sessions per user

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        logging.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logging.error(f"Failed to upload file: {e}")
        raise

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q')
    user_id = request.args.get('id')

    if not query or not user_id:
        return jsonify({"error": "Please provide both query and id parameters."}), 400

    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=25)  # Stores the last 25 messages
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # Add the user query to history
    history.append(f"User: {query}")
    response = chat_session.send_message(query)
    # Add the bot response to history
    history.append(f"Bot: {response.text}")

    return jsonify({"response": response.text})

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"/tmp/{file.filename}"
    
    # Ensure the /tmp/ directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        file.save(file_path)
        logging.info(f"File saved at {file_path}")

        uploaded_file = upload_to_gemini(file_path, mime_type=file.mimetype)
        logging.info(f"Uploaded file URI: {uploaded_file.uri}")

        chat_session = model.start_chat(
            history=[f"Analyze the image at {uploaded_file.uri}"]
        )
        response = chat_session.send_message(f"Analyze the image at {uploaded_file.uri}")

        return jsonify({"response": response.text})

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return jsonify({"error": "Failed to process the file."}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "https://your-app-url/ping"  # Replace with your actual URL
    while True:
        time.sleep(600)  # Ping every 15 minutes
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logging.info("Ping successful")
            else:
                logging.error(f"Ping failed with status code {response.status_code}")
        except Exception as e:
            logging.error(f"Ping failed with exception {e}")

if __name__ == '__main__':
    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
