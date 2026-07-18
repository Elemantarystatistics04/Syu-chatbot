# NEXUS AI — Machine Learning Chatbot

Welcome to your very own Machine Learning Chatbot! This bot doesn't rely on ChatGPT or any external APIs. Instead, it uses a Neural Network (built with scikit-learn) trained on your own custom data.

## 🚀 How to Start the Chatbot

Whenever you want to use the chatbot, follow these two steps:

1. **Start the Server**: Open your terminal (or command prompt), navigate to this folder, and run:
   ```bash
   python app.py
   ```
   *Note: As long as this terminal window is open and running, your chatbot is active.*

2. **Open the Web UI**: Open any web browser (like Chrome, Edge, or Firefox) and go to:
   **http://localhost:5000**

---

## 🧠 How to Teach the Bot New Things

The best part of this bot is that you can teach it anything! The bot's brain comes from the `intents.json` file.

### Step 1: Add Data
Open `intents.json`. You'll see blocks of data that look like this:
```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey"],
  "responses": ["Hello! I'm NEXUS.", "Hey there!"]
}
```
- **tag**: The category of the question (e.g., "weather", "about_me", "math").
- **patterns**: Different ways a user might ask the question.
- **responses**: Different ways the bot can answer (it will pick one randomly).

You can add as many new blocks as you want!

### Step 2: Retrain the Model
After saving your changes to `intents.json`, you need to retrain the neural network so it learns the new data.
1. Stop the running server (Click in the terminal and press `Ctrl+C`).
2. Run the training script:
   ```bash
   python train.py
   ```
   *You'll see it say "Training complete!" and show an accuracy percentage.*

### Step 3: Restart the Server
Now run the server again with the newly trained brain:
```bash
python app.py
```
Refresh your browser, and the bot will now know the new information!

---

## 📁 What Does Each File Do?

- **`intents.json`**: Your training data. This is where you add new knowledge.
- **`train.py`**: The script that reads `intents.json` and trains the Machine Learning model. Run this whenever you change the data.
- **`app.py`**: The web server. It connects the trained model to the website.
- **`chat.py`**: The logic that processes user messages and predicts the right response.
- **`templates/index.html`**: The beautiful user interface you see in your browser.
- **`.pkl files`**: These are the saved "brain" files created after training.

Enjoy building and expanding your AI!
