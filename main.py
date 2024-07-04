import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import logging
import base64
from PIL import Image
import io

app = Flask(__name__)

# লগিং কনফিগার করা
logging.basicConfig(level=logging.INFO)

# পরিবেশ ভেরিয়েবল থেকে API কী পাওয়া
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY পরিবেশ ভেরিয়েবল সেট করা হয়নি")

# Gemini API ইনিশিয়ালাইজ করা
genai.configure(api_key=GEMINI_API_KEY)

# চ্যাট সেশনগুলি একটি ডিকশনারিতে সংরক্ষণ করা
chat_sessions = {}

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q')
    user_id = request.args.get('id')

    if not query or not user_id:
        return jsonify({"error": "অনুগ্রহ করে উভয় প্রশ্ন এবং আইডি প্যারামিটার প্রদান করুন।"}), 400

    # একটি নতুন চ্যাট সেশন তৈরি করা যদি না থাকে
    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=25)
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # ব্যবহারকারীর প্রশ্ন ইতিহাসে যোগ করা
    history.append(f"User: {query}")
    try:
        response = chat_session.send_message(query)
        # বটের উত্তর ইতিহাসে যোগ করা
        history.append(f"Bot: {response.text}")
        return jsonify({"response": response.text})
    except Exception as e:
        logging.error(f"প্রশ্ন প্রক্রিয়াকরণে ত্রুটি: {e}")
        return jsonify({"error": "প্রশ্ন প্রক্রিয়াকরণ ব্যর্থ হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।"}), 500

@app.route('/describe_image', methods=['POST'])
def describe_image():
    if 'image' not in request.files:
        return jsonify({"error": "অনুগ্রহ করে একটি ইমেজ ফাইল আপলোড করুন।"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file.stream)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # জেমিনি API-তে ইমেজ পাঠানো এবং বর্ণনা প্রাপ্তি
        response = genai.describe_image(image_data=img_str)
        description = response['description']

        return jsonify({"description": description})
    except Exception as e:
        logging.error(f"ইমেজ প্রক্রিয়াকরণে ত্রুটি: {e}")
        return jsonify({"error": "ইমেজ প্রক্রিয়াকরণ ব্যর্থ হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।"}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "https://symmetrical-journey-6at4.onrender.com/ping"  # আপনার অ্যাপের URL দিয়ে প্রতিস্থাপন করুন
    while True:
        time.sleep(600)  # প্রতি ১০ মিনিটে পিং করা
        try:
            requests.get(url)
        except requests.RequestException as e:
            logging.error(f"কিপ-অ্যালাইভ পিং ব্যর্থ হয়েছে: {e}")

if __name__ == '__main__':
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
