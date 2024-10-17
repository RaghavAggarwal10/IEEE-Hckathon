from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline
)

import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables
model_emb = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
tokenizer_t5 = T5Tokenizer.from_pretrained("valhalla/t5-base-e2e-qg", legacy=False)
model_t5 = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-e2e-qg")

# Helper functions
def get_pdf_text_with_pages(pdf_docs):
    text_with_pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text_with_pages.append((text, page_num + 1))  # Store text and corresponding page number
    return text_with_pages

def get_text_chunks_with_pages(text_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjust chunk size as needed
    chunks_with_pages = []
    
    for text, page_num in text_with_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_num))  # Store chunk with corresponding page number
    
    return chunks_with_pages

def get_vector_store_with_pages(chunks_with_pages):
    embeddings = []
    page_numbers = []

    # Create embeddings and store page numbers
    for chunk, page_num in chunks_with_pages:
        embeddings.append(model_emb.encode(chunk))
        page_numbers.append(page_num)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    dimension = embeddings.shape[1]
    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(embeddings)

    # Save FAISS index and page numbers
    faiss.write_index(vector_store, "faiss_index.index")
    np.save("page_numbers.npy", np.array(page_numbers))  # Save page numbers

    return vector_store

def get_conversational_chain():
    # Authenticate using your Hugging Face token
    login(token=os.getenv("HF_TOKEN"))  # Use your Hugging Face token

    # Set up a pipeline for text generation
    pipe = pipeline(
        "text-generation", 
        model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"), 
        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=500 # Set the max number of tokens to be generated
    )

    # Integrate the Hugging Face pipeline into LangChain
    hf_model = HuggingFacePipeline(pipeline=pipe)

    # Define the prompt template for question answering
    prompt_template1 = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context". Do not provide an incorrect answer.

    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    # Create a PromptTemplate
    prompt = PromptTemplate(template=prompt_template1, input_variables=["context", "question"])

    # Load the question answering chain using the Hugging Face model
    chain = load_qa_chain(hf_model, chain_type="stuff", prompt=prompt)

    return chain

def question_suggestion(input_text):
    # Tokenize the input text with truncation to max length
    max_input_length = 512
    input_ids = tokenizer_t5(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).input_ids

    # Generate questions with adjusted parameters to avoid truncation
    with torch.no_grad():
        output = model_t5.generate(
            input_ids, 
            num_return_sequences=3,        # Generate 3 questions
            num_beams=5,                   # Use beam search for better diversity
            max_length=64,                 # Increase max_length to avoid truncation
            no_repeat_ngram_size=3,        # Avoid repetition in questions
            early_stopping=True
        )

    # Decode the generated output to text
    generated_questions = [tokenizer_t5.decode(o, skip_special_tokens=True) for o in output]

    # Clean the questions
    cleaned_questions = []
    for idx, question in enumerate(generated_questions):
        split_questions = question.split("<sep>")  # Split by the <sep> token
        for q in split_questions:
            q = q.strip()
            if q and q not in cleaned_questions:  # Avoid empty or duplicate questions
                cleaned_questions.append(q)

    return cleaned_questions

def user_input_with_citations(user_question, chunks_with_pages):
    # Load FAISS index and page numbers
    vector_store = faiss.read_index("faiss_index.index")
    page_numbers = np.load("page_numbers.npy")

    # Create embedding for the user's question
    user_embedding = model_emb.encode([user_question])

    # Perform similarity search
    distances, indices = vector_store.search(np.array(user_embedding, dtype=np.float32), k=5)
    
    # Retrieve top matching text chunks and their page numbers
    docs = []
    cited_chunks = []
    cited_pages = []

    for idx in indices[0]:
        chunk, page_num = chunks_with_pages[idx]
        docs.append(chunk)  # Only store the chunk content
        cited_chunks.append(chunk)  # Keep chunk for citation
        cited_pages.append(page_num)

    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Run the question-answering model
    response = chain(
        {"input_documents": [Document(page_content=chunk) for chunk in docs], "question": user_question},
        return_only_outputs=True
    )

    # Format the final response with the exact citation
    output_text = response['output_text']
    citation = f"Cited from: Page {cited_pages[0]}, Line: '{cited_chunks[0][:100]}...'"
    final_response = f"{output_text}\n\n{citation}"
    
    return final_response

@app.route("/upload", methods=["POST"])
def upload_pdf_and_query():
    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files provided"}), 400

    pdf_docs = request.files.getlist("pdfs")
    user_question = request.form.get("question", "")

    # Step 2: Extract text from the PDFs with page numbers
    text_with_pages = get_pdf_text_with_pages(pdf_docs)
    
    # Step 3: Split the text into chunks while preserving page numbers
    chunks_with_pages = get_text_chunks_with_pages(text_with_pages)
    
    # Step 4: Create or load the FAISS vector store
    get_vector_store_with_pages(chunks_with_pages)  # This creates and saves the FAISS index

    # Step 5: Get the answer with citation
    answer_with_citation = user_input_with_citations(user_question, chunks_with_pages)

    # Step 6: Generate questions based on PDF content
    questions = question_suggestion(" ".join([text for text, _ in text_with_pages]))

    return jsonify({
        "answer": answer_with_citation,
        "questions": questions
    })

if __name__ == "__main__":
    app.run(debug=True)
