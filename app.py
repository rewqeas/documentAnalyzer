import streamlit as st
import google.generativeai as genai
import numpy as np
# from utils.vector_store import save_vector_store, load_vector_store, build_vector_store
from utils.document_loader import load_pdf, load_docx, load_text, chunk_text
from utils.text_embedder import embed_chunks, embed_query, retrive_similar_chunks    

import time
page_title = "Document Search App"
st.set_page_config(page_title=page_title, page_icon=":mag_right:")
st.title(page_title)    


#sidebar
st.sidebar.title("Document Search")
user_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

model_options = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
    "gemini-1.0-pro-vision",
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-2.0-flash-lite",

]
selected_model = st.sidebar.selectbox("Choose a Gemini Model", model_options, index=1)


if not user_api_key:
    st.sidebar.warning("Please enter your password to access the app.")
    st.stop()


try:
    genai.configure(api_key=user_api_key)
    chat_model = genai.GenerativeModel(selected_model)

    # ✅ Validate the API key by making a minimal test request
    test_response = chat_model.generate_content("Hello!")

    
    if hasattr(test_response, "text") and test_response.text:
        st.sidebar.success(f"✅ API key is valid. You can now use the app. The selected model is: {selected_model}")
    else:
        st.sidebar.error("❌ API key test failed. Please check your key.")
        st.stop()


except Exception as e:  
    st.sidebar.error(f"Error configuring API key: {e}")
    st.stop()


#upload the document
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])


#check if the file is uploaded
if uploaded_file:
    #if new document is uploaded, clear everthing
    if "load_uploaded_file" not in st.session_state or st.session_state.load_uploaded_file != uploaded_file.name:
        st.session_state.clear()
        st.session_state.load_uploaded_file = uploaded_file.name


    #reconfigure genai(needed after the session )
    genai.configure(api_key=user_api_key)
    chat_model = genai.GenerativeModel(selected_model)

    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "pdf":
        raw_text = load_pdf(uploaded_file)

    elif file_ext == "docx":
        raw_text = load_docx(uploaded_file)

    elif file_ext == "txt":
        raw_text = load_text(uploaded_file)

    else:
        st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        st.stop()
    status_msg_for_loading = st.info("Document loaded successfully. Processing...")
    

    chunks = chunk_text(raw_text)
    embeddings = np.array(embed_chunks(chunks))
    st.session_state['chunks'] = chunks
    st.session_state['embeddings'] = embeddings
    st.session_state['document_processor'] = uploaded_file.name
    st.session_state['document_processed'] = True

    status_msg_for_loading.success("Document processed and embeddings created successfully.")

else:
    st.session_state["document_processor"] = None
    st.info("Please upload a document to get started.")


#ask the question only after document is processed
if st.session_state.get('document_processed', False):
    st.subheader("Ask a question about the document")

    #ensure fresh key to clear old query when document is re-uploaded
    # query_key = f"query_input_{st.session_state['last_uploaded_file']}"

    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = "default_file_name"  # or None, or ""

    query_key = f"query_input_{st.session_state['last_uploaded_file']}"

    query = st.text_input("Enter your question here:", key=query_key)
    
    if query:
        status_msg = st.empty()
        status_msg.info("Searching for answers...")
        top_chunks = retrive_similar_chunks(query, 
                                            chunk_embeddings=st.session_state['embeddings'],
                                              chunks = st.session_state['chunks'])
        

        context = "\n\n".join(top_chunks)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"

        status_msg.info("Generating answer...")
        st.markdown("### Context:")
        response_area = st.empty()

        try:
            response_stream = chat_model.generate_content(prompt, stream=True)
            full_response = ""

            for chunk in response_stream:
                full_response += chunk.text
                response_area.markdown(full_response)  
                time.sleep(0.01)  # Simulate streaming delay

            status_msg.success("Answer generated successfully.")

        except Exception as e:
            st.error(f"Error generating response: {e}")

else:
    st.info("Please upload a document to ask questions.")


