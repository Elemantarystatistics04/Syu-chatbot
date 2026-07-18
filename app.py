"""
SYU AI — Flask Web Server (LLaMA 3 Cloud Edition)
==============================================
Serves the chat UI and exposes a /chat API endpoint.
"""

from flask import Flask, request, jsonify, render_template
from chat import get_response  # pyrefly: ignore

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main chat UI."""
    from flask import make_response
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Body: { "messages": [{"role": "user", "content": "..."}] }
    Returns: { "response": "bot reply" }
    """
    data = request.get_json()
    
    # We now expect an array of messages for conversation history
    if not data or 'messages' not in data:
        return jsonify({'error': 'No messages provided'}), 400

    messages = data['messages']
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({'error': 'Messages must be a non-empty array'}), 400

    # The AI inference engine (chat.py) now yields chunks
    def generate():
        import json
        for chunk in get_response(messages):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    from flask import Response
    return Response(generate(), mimetype='text/event-stream')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'agent': 'SYU AI (LLaMA 3 Cloud)', 'version': '2.0'})


if __name__ == '__main__':
    print("=" * 50)
    print("  SYU AI (LLaMA 3 Cloud Edition) — Web Server Starting")
    print("=" * 50)
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000)
