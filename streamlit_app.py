import streamlit as st
from main import query_documents, vectorize_and_store, index
import os
from werkzeug.utils import secure_filename

# Page config
st.set_page_config(
    page_title="LinChat - Stadgedokument Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for LinTek styling
st.markdown("""
<style>
    :root {
        --magenta: #E1007A;
        --purple: #7B217C;
        --cyan: #00B5E2;
    }
    
    .stApp {
        background: linear-gradient(135deg, #E1007A 0%, #7B217C 100%);
    }
    
    .main {
        background: white;
        border-radius: 16px;
        padding: 2rem;
    }
    
    h1 {
        color: #E1007A;
        font-weight: 700;
    }
    
    .stButton>button {
        background: #E1007A;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: #7B217C;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🤖 LinChat")
st.subheader("Ställ frågor om LinTeks stadgedokument")

# Sidebar for file upload and stats
with st.sidebar:
    st.header("📁 Dokumenthantering")
    
    # Stats
    try:
        stats = index.describe_index_stats()
        st.metric("Vektorer i databas", stats.total_vector_count)
    except:
        st.metric("Vektorer i databas", "0")
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader("Ladda upp PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Save file
        os.makedirs("docs", exist_ok=True)
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join("docs", filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ {filename} uppladdad!")
    
    st.divider()
    
    # Vectorize button
    if st.button("🔄 Vektorisera dokument", use_container_width=True):
        with st.spinner("Vektoriserar dokument..."):
            try:
                vectorize_and_store()
                st.success("✅ Dokument vektoriserade!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Fel: {str(e)}")
    
    st.divider()
    
    # List files
    st.subheader("Uppladdade filer")
    if os.path.exists("docs"):
        files = [f for f in os.listdir("docs") if f.endswith('.pdf')]
        if files:
            for file in files:
                st.text(f"📄 {file}")
        else:
            st.text("Inga filer ännu")

# Chat interface
st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ställ en fråga om dokumenten..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Tänker..."):
            try:
                response = query_documents(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Fel: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})