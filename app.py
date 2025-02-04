import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
GOOGLE_API_KEY = "AIzaSyAtwg_e8TSyUIWfyjg_7QI0lOCeniBkfsc"

# Set Streamlit page configuration
st.set_page_config(page_title="Chat With Multiple PDF", layout="wide")

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to define a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say, "Answer is not available in the context."
    Don't provide the context itself in your response.
    
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the QA chain with the model and prompt
    chain = load_qa_chain(model=model, prompt=prompt)
    return chain

# Function to handle user input and provide responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Perform similarity search on the database using the user's question
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Generate a response using the chain
    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )
    
    # Print and display the response
    print(response)
    st.write("Reply: ", response["output_text"])

# Main Streamlit application
def main():
    st.header("Chat with Multiple PDFs using Gemini") 
    user_question = st.text_input("Ask a Question from the PDF Files") 
    if user_question: 
        user_input(user_question) 
    with st.sidebar: 
        st.title("Menu:") 
        pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True) 
        if st.button("Submit & Process"): 
            with st.spinner("Processing..."): 
                raw_text = get_pdf_text(pdf_docs) 
                text_chunks = get_text_chunks(raw_text) 
                get_vector_store(text_chunks) 
                st.success("Done")

if __name__ == "__main__":
    main()
