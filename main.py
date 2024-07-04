from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import os

app = Flask(__name__)

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
    generation_config=generation_config,
    roles=["user", "model"]  # Adding roles here
)

chat_sessions = {}  # Dictionary to store chat sessions per user

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

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
    file.save(file_path)
    
    uploaded_file = upload_to_gemini(file_path, mime_type=file.mimetype)
    chat_session = model.start_chat(
        history=[f"Analyze the image at {uploaded_file.uri}"]
    )
    response = chat_session.send_message(f"Analyze the image at {uploaded_file.uri}")
    
    return jsonify({"response": response.text})

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
                print("Ping successful")
            else:
                print("Ping failed with status code", response.status_code)
        except Exception as e:
            print("Ping failed with exception", e)

if __name__ == '__main__':
    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
