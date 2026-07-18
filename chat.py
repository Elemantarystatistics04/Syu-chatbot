"""
SYU AI — Hybrid Chat Logic (ML Intents + LLaMA 3)
===================================================
This module first checks if a user's message matches a trained intent.
If the confidence is high, it returns a fast pre-defined response.
If the confidence is low or the intent is unknown, it falls back to
a local Ollama instance running LLaMA 3.
"""

import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import os

# ── 1. Setup & Load ML Models ──────────────────────────────────────
lemmatizer = WordNetLemmatizer()

# Attempt to load ML models (silently fail if they don't exist yet, 
# falling back entirely to LLaMA 3)
MODEL_LOADED = False
try:
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    with open('intents_data.pkl', 'rb') as f:
        intents = pickle.load(f)
    MODEL_LOADED = True
except (FileNotFoundError, EOFError):
    print("⚠️ ML models not found. Running purely on LLaMA 3 fallback.")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    if not MODEL_LOADED:
        return None, 0.0

    bow = bag_of_words(sentence, words)
    # The model expects a 2D array: [bow]
    res = model.predict_proba([bow])[0]
    
    # Get highest probability index
    max_index = np.argmax(res)
    intent_tag = classes[max_index]
    probability = res[max_index]
    
    return intent_tag, probability


# ── 2. Groq Cloud Fallback Configuration ──────────────────────────────
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Use the API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

def get_llama_response(messages):
    """Sends the conversational history to Groq (LLaMA 3 Cloud) and yields the generative response chunk by chunk."""
    system_prompt = {
        "role": "system",
        "content": "You are SYU, a highly intelligent and helpful AI assistant. You can answer general knowledge, coding, and math questions perfectly, even if you previously stated otherwise. Be concise, friendly, and structure your answers with bullet points if it's a long explanation."
    }
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[system_prompt] + messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                
    except Exception as e:
        yield f"⚠️ An error occurred with Groq: {str(e)}"


# ── 3. Main Hybrid Chat Function ───────────────────────────────────
def get_response(messages):
    """
    1. Checks latest user message against ML Intent model.
    2. If confidence > 75% and not unknown, use Intents (yields once).
    3. Else fallback to Ollama stream.
    """
    if not messages:
        yield "I didn't receive a message!"
        return
        
    latest_user_message = messages[-1].get("content", "")
    
    # Predict Intent
    intent_tag, prob = predict_class(latest_user_message)
    
    # Threshold for fast intent responses
    if intent_tag and prob > 0.75 and intent_tag != 'unknown':
        # Find the appropriate response in intents.json
        for i in intents['intents']:
            if i['tag'] == intent_tag:
                yield random.choice(i['responses'])
                return
                
    # Fallback to Ollama (yields chunk by chunk)
    for chunk in get_llama_response(messages):
        yield chunk


# ── Test Mode ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("SYU AI — Hybrid Test Mode")
    print("Type 'quit' to exit\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("SYU: Goodbye!")
            break
        if not user_input:
            continue
            
        chat_history.append({"role": "user", "content": user_input})
        print("\nThinking...\n")
        
        reply = ""
        print("SYU: ", end="", flush=True)
        for chunk in get_response(chat_history):
            print(chunk, end="", flush=True)
            reply += chunk
        print("\n")
        
        chat_history.append({"role": "assistant", "content": reply})
