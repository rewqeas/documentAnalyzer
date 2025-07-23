import streamlit as st
import google.generativeai as genai

from utils.vector_store import save_vector_store, load_vector_store, build_vector_store
from utils.document_loader import load_pdf, load_docx, load_text, chunk_text    

import time
page_title = "Document Search App"
st.set_page_config(page_title=page_title, page_icon=":mag_right:")
st.title(page_title)    


#sidebar
st.sidebar.title("Document Search")
user_api_key = st.sidebar.text_input("enter your Gemini Api key", type="password")

if not user_api_key:
    st.sidebar.warning("Please enter your password to access the app.")
    st.stop()


try:
    genai.configure(api_key=user_api_key)
    chat_model = genai.GenerativeModel("gemini-2.0-flash-lite")

    # test the key by sending a tiny request
    response = chat_model.generate_content("Hello")

    st.sidebar.success("API key is valid. You can now use the app.")

    
except Exception as e:  
    st.sidebar.error(f"Error configuring API key: {e}")
    st.stop()
