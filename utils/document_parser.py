import os
import requests
import tempfile
import fitz  # PyMuPDF
import docx

def download_file(url: str) -> str:
    """
    Downloads a document from a URL and saves it to a temp file.
    Returns the local file path.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file from: {url}")
    
    # Pick extension
    if ".pdf" in url:
        suffix = ".pdf"
    elif ".docx" in url:
        suffix = ".docx"
    else:
        raise ValueError("Unsupported file type (must be .pdf or .docx)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def extract_text_from_pdf(path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(path: str) -> str:
    """
    Extracts text from a DOCX file.
    """
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_document_from_url(url: str) -> str:
    """
    Main wrapper: download file, detect type, extract text, clean up temp file.
    """
    path = download_file(url)
    try:
        if path.endswith(".pdf"):
            return extract_text_from_pdf(path)
        elif path.endswith(".docx"):
            return extract_text_from_docx(path)
    finally:
        os.remove(path)
