import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import os

# Load the environment variables from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

import random

def generate_random_questions(text_chunks, num_questions=5):
    question_types = [
        "What is the main idea of the following text?",
        "Can you explain the key concept described here?",
        "What are the implications of the information provided below?",
        "How does the following text contribute to the overall context?",
        "What details stand out in this text?",
    ]
    
    random_questions = []
    for _ in range(num_questions):
        chunk = random.choice(text_chunks)  # Select a random chunk
        question_template = random.choice(question_types)  # Pick a random question type
        question = f"{question_template}\n\n{chunk[:300]}..."  # Add a snippet from the text
        random_questions.append(question)
    
    return random_questions


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "answer is not available in the context".\n\n
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GEMINI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_answers_for_questions(questions, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain()
    answers = []
    for question in questions:
        docs = vector_store.similarity_search(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answers.append(response["output_text"])
    return answers

def main():
    st.set_page_config(page_title="Random Q&A from PDFs", layout="wide")
    st.header("Generate Random Questions and Answers from PDF Content")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing uploaded files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed and vector store updated!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main content
    if os.path.exists("faiss_index"):
        st.subheader("Generate Random Questions and Answers")
        num_questions = st.number_input("How many random questions do you want to generate?", min_value=1, step=1, value=5)
        if st.button("Generate Questions"):
            text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
            random_questions = generate_random_questions(text_chunks, num_questions)
            st.session_state["questions"] = random_questions
            st.session_state["answers"] = get_answers_for_questions(random_questions, text_chunks)
            st.success("Questions generated successfully!")

        if "questions" in st.session_state:
            st.write("### Random Questions")
            for i, question in enumerate(st.session_state["questions"]):
                st.write(f"**Question {i+1}:** {question}")
            
            if st.button("Show Answers"):
                st.write("### Answers")
                for i, answer in enumerate(st.session_state["answers"]):
                    st.write(f"**Answer {i+1}:** {answer}")
            else:
                st.warning("Answers are hidden. Press 'Show Answers' to reveal them.")
    else:
        st.warning("No PDFs have been processed. Please upload and process PDFs first.")

if __name__ == "__main__":
    main()
