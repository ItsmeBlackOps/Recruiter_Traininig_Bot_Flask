#app.py
from flask import Flask, request, jsonify
import pdfplumber
from database import QuestionAnswerBot  # Import your bot class here
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load PDF content once at the start
pdf_path = "1678899842229.pdf"  # Replace with the path to your PDF file
pdf_content = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        pdf_content += page.extract_text() or ""

# Initialize your bot
qa_bot = QuestionAnswerBot(pdf_content)

@app.route('/', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = qa_bot.generate_answer(question)
    print(answer)
    return answer

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False in a production environment
