from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

docs = []

# Load from URL and split documents
url = [
    "https://developer.hashicorp.com/nomad/docs/upgrade/upgrade-specific",
    "https://developer.hashicorp.com/nomad/docs/release-notes/nomad/v1_8_x",
    "https://developer.hashicorp.com/nomad/docs/release-notes/nomad/v1_9_x",
    "https://developer.hashicorp.com/nomad/docs/release-notes/nomad/v1-10-x",
    "https://developer.hashicorp.com/nomad/docs/release-notes/nomad/upcoming",
] 
docs.extend(WebBaseLoader(url).load())

# Load from PDF and split documents
# pdf_path = "document.pdf"
# docs.extend(PyPDFLoader(pdf_path).load())

# # Load from text file and split documents
# text_path = "notes.txt"
# docs.extend(TextLoader(text_path).load())

# Split documents into chunks for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embed and store vectors
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Updated class
vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")

# Ask a question
query = "As a support employee at Hashicorp I got the following question and needs to answer to customers with a clear and detailed way: `I am running Nomad version 1.9.3 and want to update to 1.10.0, are there any breaking changes/deprecation/major changes I should be aware of? Can you provide me with a detailed list of the breaking changes?`"
relevant_docs = vectorstore.similarity_search(query, k=5)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Generate answer with Ollama
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
print(response["message"]["content"])
