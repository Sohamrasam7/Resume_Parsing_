import re
import torch
import PyPDF2
import string
from fastapi import FastAPI, File, UploadFile
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@app.post("/similarity_score")
async def get_similarity_score(resume: UploadFile = File(...), job_description: UploadFile = File(...)):
    resume_pdf_reader = PyPDF2.PdfReader(resume.file)
    job_description_pdf_reader = PyPDF2.PdfReader(job_description.file)

    resume_text = ''
    for page in range(len(resume_pdf_reader.pages)):
        page_obj = resume_pdf_reader.pages[page]
        resume_text += page_obj.extract_text()
    resume_pdf_reader

    job_description_text = ''
    for page in range(len(job_description_pdf_reader.pages)):
        page_obj_new = job_description_pdf_reader.pages[page]
        job_description_text += page_obj_new.extract_text()
    job_description_pdf_reader

    # Data Preprocessing:
    def preprocess_text(text):
       # Convert all text to lowercase
       text = text.lower()
       # Remove all non-alphanumeric characters, except for spaces
       text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
       # Remove all punctuation marks
       text = text.translate(str.maketrans('', '', string.punctuation))
       # Split text into individual words
       words = text.split()
       # Filter out words that are less than 2 characters long
       words = [word for word in words if len(word) >= 2]
       resume_text=re.sub('\n', ' ', resume_text)
       # Remove all extra spaces
       text = re.sub(r'\s+', ' ', text)
       return text

    # Encode the text using BERT
    presprocess_resume_text=preprocess_text(resume_text)
    preprocess_job_description_text=preprocess_text(job_description_text)

    resume_tokens = tokenizer.encode(presprocess_resume_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')
    job_description_tokens = tokenizer.encode(preprocess_job_description_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

    resume_input_ids = torch.tensor(resume_tokens).unsqueeze(0)
    job_description_input_ids = torch.tensor(job_description_tokens).unsqueeze(0)

    resume_outputs = model(resume_input_ids)
    job_description_outputs = model(job_description_input_ids)

    resume_encoded = resume_outputs[1].detach().numpy()
    job_description_encoded = job_description_outputs[1].detach().numpy()

    # Calculate cosine similarity
    similarity_score = cosine_similarity(resume_encoded, job_description_encoded)[0][0]

    # Output the similarity score
    return {"similarity_score": float("{:.2f}".format(similarity_score))*100}