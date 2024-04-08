# 支持word  pdf  txt等格式的问答
# https://github.com/gabacode/chatPDF
# https://github.com/linjungz/chat-with-your-doc
# 
import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.callbacks.base import BaseCallbackHandler

from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader, CSVLoader, MWDumpLoader)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage,AIMessage


os.environ['OPENAI_API_KEY'] = ''

files = [
    './data/0.docx',
    './data/1.pdf',
    './data/2.docx',
    './data/3.docx'
]
docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)



for file in files:
    print(f"Loading file: {file}")
    ext_name = os.path.splitext(file)[-1]
    if ext_name == ".pptx":
        loader = UnstructuredPowerPointLoader(file)
    elif ext_name == ".docx":
        loader = UnstructuredWordDocumentLoader(file)
    elif ext_name == ".pdf":
        loader = PyPDFLoader(file)
    elif ext_name == ".csv":
        loader = CSVLoader(file_path=file)
    elif ext_name == ".xml":
        loader = MWDumpLoader(file_path=file, encoding="utf8")
    else:
        # process .txt, .html
        loader = UnstructuredFileLoader(file)

    doc = loader.load_and_split(text_splitter)            
    docs.extend(doc)
    print("Processed document: " + file)







chat = ChatOpenAI(
    temperature=0,
    openai_api_key='',
    request_timeout=60,
    model="gpt-4-0125-preview",  # Model name is needed for openai.com only
    streaming=False,
)