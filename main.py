from pinecone import Pinecone, ServerlessSpec 
from openai import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import hashlib
import time

load_dotenv()

pc = Pinecone(api_key=os.getenv("pinecone_api_key"))
client = OpenAI(api_key=os.getenv("openai_api_key"))

# Create or connect to index
INDEX_NAME = "association-docs"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def load_pdfs(docs_folder="docs"):
    """Load all PDFs from docs folder"""
    texts = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    texts.append({
                        "text": text,
                        "source": filename,
                        "page": page_num + 1
                    })
    return texts

def chunk_text(text, chunk_size=500):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def sanitize_id(text):
    """Create ASCII-safe ID from text"""
    return hashlib.md5(text.encode()).hexdigest()

def create_embedding_with_retry(text, max_retries=3):
    """Create embedding with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)[:100]}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def vectorize_and_store():
    """Load PDFs, create embeddings, and store in Pinecone"""
    print("Loading PDFs...")
    docs = load_pdfs()
    print(f"Found {len(docs)} pages to process")
    
    vectors = []
    
    for idx, doc in enumerate(docs):
        print(f"Processing {doc['source']} - Page {doc['page']}...")
        chunks = chunk_text(doc["text"])
        
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():  # Skip empty chunks
                continue
                
            # Create embedding with retry
            embedding = create_embedding_with_retry(chunk)
            
            # Prepare vector for Pinecone with sanitized ID
            vector_id_base = f"{doc['source']}_p{doc['page']}_c{chunk_idx}"
            vector_id = sanitize_id(vector_id_base)
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"]
                }
            })
    
    # Upsert to Pinecone in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i//batch_size + 1} ({len(batch)} vectors)")
    
    print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")

def query_documents(question, top_k=3):
    """Query the vector database and get response from ChatGPT"""
    # Create embedding for question
    response = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding
    
    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Build context from results
    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Query ChatGPT with context
    chat_response = client.chat.completions.create(
        #model="gpt-4",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about association governing documents. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    print(chat_response.usage)

    return chat_response.choices[0].message.content

def main():
    """Main function for terminal interaction"""
    print("=== Association Document Chatbot ===")
    print("Commands:")
    print("  'upload' - Upload documents from docs folder")
    print("  'quit' or 'exit' - Exit the chatbot")
    print("  Or just type your question!\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if user_input.lower() == 'upload':
            print("Uploading documents to Pinecone...")
            try:
                vectorize_and_store()
                print("Upload complete!")
            except Exception as e:
                print(f"Error uploading documents: {e}")
            continue
        
        # Handle question
        print("Thinking...")
        try:
            answer = query_documents(user_input)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()

