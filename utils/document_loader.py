import fitz  # PyMuPDF
from docx import Document #pip install python-docx

def load_pdf(file):
    doc = fitz.open(file)#open the pdf file
    text = ""

    for page in doc:
        text += page.get_text()
    return text

def load_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])#  a list which takes a paragraph as a element

def load_text(file):
    return file.read().decode('utf-8') #convert the paragraph to utf-8 format 


def chunk_text(text, chunk_size = 300, overlap = 50):
    """
    Splits the text into chunks of specified size with a specified overlap.
    
    :param text: The text to be chunked.
    :param chunk_size: The size of each chunk.
    :param overlap: The number of characters to overlap between chunks.
    :return: A list of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks