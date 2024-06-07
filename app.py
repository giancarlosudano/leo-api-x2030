from flask import Flask, request, jsonify

app = Flask(__name__)

# Definisci una chiave di autenticazione fissa
API_KEY = "secret"

@app.route('/ChatCompletion', methods=['POST'])
def chat_completion():
    data = request.json
    question = data.get('question')
    key = data.get('key')

    if key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    # Simula una risposta alla domanda
    response = {
        'question': question,
        'answer': 'This is a sample response to your question.'
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
