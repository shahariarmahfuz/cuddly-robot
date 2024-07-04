import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

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
)

chat_sessions = {}  # Dictionary to store chat sessions per user

# Load pre-trained MobileNetV2 model
image_model = MobileNetV2(weights='imagenet')

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q')
    user_id = request.args.get('id')

    if not query or not user_id:
        return jsonify({"error": "অনুগ্রহ করে প্রশ্ন এবং ইউজার আইডি প্রদান করুন।"}), 400

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

@app.route('/analyze_image', methods=['GET'])
def analyze_image():
    user_id = request.args.get('id')
    question = request.args.get('q')
    image_url = request.args.get('image_url')

    if not user_id or not question or not image_url:
        return jsonify({"error": "অনুগ্রহ করে ইউজার আইডি, প্রশ্ন এবং ইমেজ URL প্রদান করুন।"}), 400

    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=25)  # Stores the last 25 messages
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # Download the image from the provided URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    except Exception as e:
        return jsonify({"error": f"ইমেজ ডাউনলোড করতে ব্যর্থ হয়েছে: {str(e)}"}), 400

    # Analyze the image using TensorFlow model
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = image_model.predict(x)
    labels = decode_predictions(preds, top=3)[0]
    image_description = ', '.join([label[1] for label in labels])

    # Combine image description with the user's question
    combined_query = f"The image contains: {image_description}. {question}"

    # Add the user query to history
    history.append(f"User: {combined_query}")
    bot_response = chat_session.send_message(combined_query)
    # Add the bot response to history
    history.append(f"Bot: {bot_response.text}")

    return jsonify({"response": bot_response.text})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "https://your-deployed-url/ping"  # Replace with your actual URL
    while True:
        time.sleep(600)  # Ping every 10 minutes
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
