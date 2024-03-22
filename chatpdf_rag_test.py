# 支持word  pdf  txt等格式的问答
# https://github.com/gabacode/chatPDF
# https://github.com/linjungz/chat-with-your-doc
# 
import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.callbacks.base import BaseCallbackHandler

from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader, CSVLoader, MWDumpLoader)
import langchain.text_splitter as text_splitter
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)




files = [
    './data/0.docx',
    './data/1.pdf',
    './data/2.docx',
    './data/3.docx'
]
docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(
        deployment='"text-embedding-ada-002"',
        chunk_size=1
        ) # type: ignore
embeddings = HuggingFaceEmbeddings(
    model_name=(
        'sentence-transformers/'
        'multi-qa-MiniLM-L6-cos-v1'
    )
)


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

print("Generating embeddings and ingesting to vector db.")
vector_db = FAISS.from_documents(docs, embeddings)
print("Vector db initialized.")
FAISS.save_local(vector_db, 'path', 'index_name')
print("Vector db saved to local")










class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


streaming=False
answer_container = None
condense_question_container = None
chatgpt = ChatOpenAI(
        temperature=0,
        request_timeout=10,
        model="gpt-4-0125-preview",  # Model name is needed for openai.com only
        streaming=streaming,
        callbacks=[StreamHandler(answer_container)] if streaming else []
    ) # type: ignore

if streaming:
    condense_question_llm = ChatOpenAI(
        temperature=0,
        request_timeout=50,
        streaming=True,
        model="gpt-4-0125-preview",
        callbacks=[StreamHandler(condense_question_container, "🤔...")]
    ) # type: ignore
else:
    condense_question_llm = chatgpt


# stuff chain_type seems working better than others
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""1.pdf is a good case,2.docx is a medium case,3.docx is a poor case, 0.docx is an evaluation standard.
    Chat History:
    {chat_history}

    Follow Up Input:
    {question}

    Standalone Question:"""
    )            
chatchain = ConversationalRetrievalChain.from_llm(llm=chatgpt, 
                                        retriever=vector_db.as_retriever(k=4),
                                        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                        condense_question_llm=condense_question_llm,
                                        chain_type='stuff',
                                        return_source_documents=True,
                                        verbose=False)
                                        # combine_docs_chain_kwargs=dict(return_map_steps=False))


ch = []
message = '三个文档的信息'

result = chatchain({
        "question": message,
        "chat_history": ch
},
return_only_outputs=True)
print(result['answer'])

"""
ProjectIdearubric.docx的内容是对论文的评估指标，1.pdf的内容是好的案例，2.docx是中等案例，3.docx是差等案例，请对之后的文档按照这评估指标以及三个等级来评估，给出得分和三个等级
"""