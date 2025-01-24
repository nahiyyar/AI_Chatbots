from flask import Flask, request, jsonify, render_template, send_file
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import json
import os
import re
import spacy
import torch

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
chatbot_model = SentenceTransformer('all-MiniLM-L6-v2')

questions, responses, question_embeddings = [], [], None
chatbot_name = "Bot"

def parse_pdf_to_json(pdf_path):
    def generate_tag(pattern):
        """Generate a tag from a question."""
        doc = nlp(pattern)
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return "_".join(keywords[:3])

    def clean_question(question):
        """Clean a question by removing numbering or symbols."""
        match = re.match(r"^\d+[\).\-\s]+(.*)", question)
        return match.group(1).strip() if match else question.strip()

    def is_question_line(line):
        """Check if a line is likely a question."""
        return bool(re.match(r"^\d+[\).\s]+[A-Za-z]", line)) or line.strip().endswith("?")

    def is_header_or_footer(line):
        """Filter out headers and footers based on patterns."""
        patterns = [r"^Frequently Asked Questions$", r"^\d+$", r"^http[s]?://.*"]
        return any(re.match(pattern, line.strip()) for pattern in patterns)

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                filtered_lines = [
                    line.strip() for line in page_text.split("\n")
                    if not is_header_or_footer(line) and line.strip()
                ]
                lines.extend(filtered_lines)

    qa_pairs = []
    current_question, current_answer, question_buffer = "", [], ""

    for line in lines:
        if is_question_line(line):
            if current_question and current_answer:
                qa_pairs.append((current_question.strip(), " ".join(current_answer).strip()))
            question_buffer += " " + clean_question(line)
            if line.strip().endswith("?"):
                current_question, question_buffer = question_buffer.strip(), ""
            current_answer = []
        elif current_question:
            current_answer.append(line.strip())

    if current_question and current_answer:
        qa_pairs.append((current_question.strip(), " ".join(current_answer).strip()))

    predefined_intents = [
        {
            "tag": "greetings",
            "patterns": ["Hello", "Hi", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello! How can I help you?", "Hi there! Ask me anything."],
        },
        {
            "tag": "goodbye",
            "patterns": ["Goodbye", "Bye", "See you later", "Take care"],
            "responses": ["Goodbye! Have a great day.", "See you later!"],
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "I appreciate it"],
            "responses": ["You're welcome!", "Happy to help!"],
        },
    ]

    faq_intents = [
        {
            "tag": generate_tag(question),
            "patterns": [question],
            "responses": [answer]
        }
        for question, answer in qa_pairs
    ]

    all_intents = predefined_intents + faq_intents

    output_file = os.path.join(UPLOAD_FOLDER, "intents.json")
    with open(output_file, "w") as f:
        json.dump({"intents": all_intents}, f, indent=4)

    return {"intents": all_intents, "file_path": output_file}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global questions, responses, question_embeddings

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('index.html', error="No file uploaded.")
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext != '.pdf':
            return render_template('index.html', error="Only PDF files are supported.")
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        data = parse_pdf_to_json(file_path)

        questions, responses = [], []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                questions.append(pattern)
                responses.append(intent['responses'][0])

        question_embeddings = chatbot_model.encode(questions, convert_to_tensor=True)

        embeddings_file = os.path.join(UPLOAD_FOLDER, "model_embeddings.pt")
        torch.save(question_embeddings, embeddings_file)

        return render_template('chat.html', chatbot_name=chatbot_name)

    return render_template('index.html')

@app.route('/download_model', methods=['GET'])
def download_model():
    file_path = os.path.join(UPLOAD_FOLDER, "model_embeddings.pt")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "Model file not found."}), 404

@app.route('/chat', methods=['POST'])
def chat():
    global questions, responses, question_embeddings

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"response": "Please ask a question."}), 400

    query_embedding = chatbot_model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)
    best_score = scores.max()

    if best_score < 0.5:  
        return jsonify({"response": "I'm sorry, I couldn't find a relevant answer."})
    
    best_match_idx = scores.argmax()
    return jsonify({"response": responses[best_match_idx]})

if __name__ == '__main__':
    app.run(debug=True)
