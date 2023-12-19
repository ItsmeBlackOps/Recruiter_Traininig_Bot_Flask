#database.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pdfplumber


class QuestionAnswerBot:
    def __init__(self, context, chunk_size=200):
        self.chunk_size = chunk_size
        self.context_chunks = self.chunk_context(context)
        self.vectorizer = TfidfVectorizer()
        self.context_vectors = self.vectorizer.fit_transform(self.context_chunks)

    def chunk_context(self, context):
        sentences = context.split('.')
        chunks = [' '.join(sentences[i:i + self.chunk_size]) for i in range(0, len(sentences), self.chunk_size)]
        return chunks

    def find_relevant_chunk(self, question):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.context_vectors)
        most_relevant_chunk_index = np.argmax(similarities)
        return self.context_chunks[most_relevant_chunk_index]

    def generate_answer(self, question):
        relevant_chunk = self.find_relevant_chunk(question)

        genai.configure(api_key="AIzaSyDLQehxg9kSK5MqCI1N0GiDrcT9Re-XV-c")

        generation_config = {
            "temperature": 0.05,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 20480,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            
        ]
        model = genai.GenerativeModel(model_name="gemini-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        prompt_parts = [relevant_chunk, question]
        response = model.generate_content(prompt_parts)

        if self.is_response_satisfactory(response.text):
            return response.text
        else:
            return "I'm sorry, I don't have enough information to answer that question accurately."

    def is_response_satisfactory(self, response):
        if len(response) < 30 or "I don't know" in response:
            return False
        return True


def read_pdf(pdf_path):
    pdf_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text() or ""
    return pdf_content


def main():
    print("Bench Sales Recruiter Training Bot")
    print("Please provide the path to the PDF file:")
    pdf_path = input("PDF Path: ")
    pdf_content = read_pdf(pdf_path)

    qa_bot = QuestionAnswerBot(pdf_content)

    while True:
        question = input("Ask a question (type 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break

        answer = qa_bot.generate_answer(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
