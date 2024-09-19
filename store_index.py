from src.helper import load_pdf, text_split,download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PC
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embedding =download_hugging_face_embeddings()


from pinecone import Pinecone
pc= Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index('medical-bot')
docsearch=PC.from_texts([t.page_content for t in text_chunks],embedding,index_name='medical-bot')
