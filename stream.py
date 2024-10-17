import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
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

# Initialize global models
model_emb = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
tokenizer_t5 = T5Tokenizer.from_pretrained("valhalla/t5-base-e2e-qg", legacy=False)
model_t5 = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-e2e-qg")

def get_pdf_text_with_pages(pdf_docs):
    text_with_pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text_with_pages.append((text, page_num + 1))
    return text_with_pages

def get_text_chunks_with_pages(text_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_with_pages = []
    
    for text, page_num in text_with_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_num))
    
    return chunks_with_pages

def get_vector_store_with_pages(chunks_with_pages):
    embeddings = []
    page_numbers = []

    for chunk, page_num in chunks_with_pages:
        embeddings.append(model_emb.encode(chunk))
        page_numbers.append(page_num)

    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(embeddings)

    faiss.write_index(vector_store, "faiss_index.index")
    np.save("page_numbers.npy", np.array(page_numbers))

    return vector_store

def get_conversational_chain():
    # Set up a pipeline for text generation
    pipe = pipeline(
        "text-generation", 
        model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"), 
        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=500
    )

    hf_model = HuggingFacePipeline(pipeline=pipe)

    prompt_template1 = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context". Do not provide an incorrect answer.

    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template1, input_variables=["context", "question"])
    chain = load_qa_chain(hf_model, chain_type="stuff", prompt=prompt)

    return chain

def question_suggestion(input_text):
    max_input_length = 512
    input_ids = tokenizer_t5(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).input_ids

    with torch.no_grad():
        output = model_t5.generate(
            input_ids, 
            num_return_sequences=3,
            num_beams=5,
            max_length=64,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    generated_questions = [tokenizer_t5.decode(o, skip_special_tokens=True) for o in output]

    cleaned_questions = []
    for idx, question in enumerate(generated_questions):
        split_questions = question.split("<sep>")
        for q in split_questions:
            q = q.strip()
            if q and q not in cleaned_questions:
                cleaned_questions.append(q)

    return cleaned_questions

def user_input_with_citations(user_question, chunks_with_pages):
    vector_store = faiss.read_index("faiss_index.index")
    page_numbers = np.load("page_numbers.npy")

    user_embedding = model_emb.encode([user_question])

    distances, indices = vector_store.search(np.array(user_embedding, dtype=np.float32), k=5)
    
    docs = []
    cited_chunks = []
    cited_pages = []

    for idx in indices[0]:
        chunk, page_num = chunks_with_pages[idx]
        docs.append(chunk)
        cited_chunks.append(chunk)
        cited_pages.append(page_num)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": [Document(page_content=chunk) for chunk in docs], "question": user_question},
        return_only_outputs=True
    )

    output_text = response['output_text']
    citation = f"Cited from: Page {cited_pages[0]}, Line: '{cited_chunks[0][:100]}...'"
    final_response = f"{output_text}\n\n{citation}"
    
    return final_response

# Streamlit App
st.title("PDF Query Application")

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Text input for the user question
user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if uploaded_files and user_question:
        # Step 2: Extract text from the PDFs with page numbers
        text_with_pages = get_pdf_text_with_pages(uploaded_files)
        st.write("Extracted text with pages:", text_with_pages)
        
        # Step 3: Split the text into chunks while preserving page numbers
        chunks_with_pages = get_text_chunks_with_pages(text_with_pages)
        st.write("Text chunks with pages:", chunks_with_pages)
        
        # Step 4: Create or load the FAISS vector store
        get_vector_store_with_pages(chunks_with_pages)

        # Step 5: Get the answer with citation
        answer_with_citation = user_input_with_citations(user_question, chunks_with_pages)

        # Step 6: Generate questions based on PDF content
        questions = question_suggestion(" ".join([text for text, _ in text_with_pages]))

        # Display the results
        st.subheader("Answer")
        st.write(answer_with_citation)
        
        st.subheader("Generated Questions")
        for q in questions:
            st.write(f"- {q}")
    else:
        st.error("Please upload PDF files and enter a question.")
