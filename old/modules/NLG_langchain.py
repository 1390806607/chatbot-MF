"""该文件定义了聊天机器人的后端类"""
# https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
import _thread as thread
import hashlib
import hmac
import json
import re
from abc import abstractmethod
from base64 import b64encode
from random import randint
from ssl import CERT_NONE as SSL_CERT_NONE
from urllib.parse import urljoin, urlparse, urlencode
from typing import List
import google.api_core.exceptions
import requests
from websocket import WebSocketApp
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.callbacks.base import BaseCallbackHandler

from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader, CSVLoader, MWDumpLoader)

from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)
from modules.utils import NLGEnum, Configs, Message  # getRFC1123, getMacAddress
import os




class NLGBase:
    """聊天机器人基类，建议在进行聊天机器人开发时继承该类"""
    try:
        import tiktoken
        tokenEncoding = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        tokenEncoding = None

    def __init__(self, nlg_type: NLGEnum, model: str, prompt: str = None):
        self.type = nlg_type  # 机器人类型
        self.model = model  # 机器人模型
        self.prompt = prompt  # 默认提示语(用于指定机器人的身份，有助于提高针对特定领域问题的效果)，优先级低于查询时传入的prompt


    @abstractmethod
    def singleQuery(self, message: str, prompt: str = None) -> str:
        """
        简单地进行单次查询

        允许为本次查询单独指定prompt，使用优先级按照 参数prompt > 全局prompt > None 的顺序
        :param message: str 本次用户输入
        :param prompt: str 提示语(用于指定机器人的身份，有助于提高针对特定领域问题的效果)
        :return: str 对本次聊天的回复内容
        """

    @abstractmethod
    def continuedQuery(self, message: str, history: list[list[str, str]], prompt: str = None):
        """
        进行带有历史记录的查询

        按照gradio官方demo chatbot_simple的开发示例，若是想实现带有历史记录的对话，无需自行实现历史记录的保留，建议的做法是在gradio中更新历史记录，每次查询时只需要调用更新后的history即可。

        大多数API要求历史记录(含prompt)的格式应当为[{"role": "user", "content": ""}, {"role": "assistant", "message":
        ""}...]，而大多数前端的历史记录格式为[[str, str]...]，因此需要在调用前进行转换。可借助historyConverter方法进行转换。
        :param message: str 本次用户输入
        :param history: List[List[str, str]...] 分别为用户输入和机器人回复(先前的)
        :param prompt: str 提示语(用于指定机器人的身份，有助于提高针对特定领域问题的效果)
        """

    @abstractmethod
    def checkConnection(self):
        """
        检查与Host的连接状态
        对于多数未设计检查连接状态的API，可参考OpenAI的做法：直接让后端回复一句简单的话，若回复成功则自然连接成功。
        """

    def converterHistory(self, history: [[str, str]], prompt: str = None) -> list[Message]:
        """
        将[[str, str]...]形式的历史记录转换为[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}...]的格式，
        使用场景是将gradio的Chatbot聊天记录格式转换为ChatGPT/ChatGLM3的聊天记录格式
        :param history: [[str, str]...] 分别为用户输入和机器人回复(先前的)
        :param prompt: str 提示语(用于指定机器人的身份，有助于提高针对特定领域问题的效果，允许为空)
        :return: [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}...]的格式的历史记录，注意，该结果不包括
        本次的用户输入，仅转换了历史记录
        """
        sessionPrompt = prompt if prompt else self.prompt
        sessionHistory = [Message(role="system", content=sessionPrompt)] if sessionPrompt else []
        for chat in history:
            sessionHistory.append(Message(role="user", content=chat[0]))
            sessionHistory.append(Message(role="assistant", content=chat[1]))
        return sessionHistory

    @staticmethod
    def lenOfMessages(history: list[Message] = None, message: str = None) -> int:
        """
        计算历史记录中，content部分的总长度，以便于判断是否超出了API的token数限制。

        若已将本次输入计入历史记录，则无需传入message

        或者可以只传入message，不传入history，仅统计本次输入的长度
        :param history: list[Message] 历史记录
        :param message: str 本次用户输入
        """
        length = 0
        if history:
            length += sum([len(message["content"]) for message in history])
        if message:
            length += len(message)
        return length

    @staticmethod
    def lenOfTokens(history: list[Message] = None, message: str = None, token_per_zh_char: float = None) -> int:
        """
        估算消息的token长度
        :param history: list[Message] 历史记录
        :param message: str 本次用户输入
        :param token_per_zh_char: float 每个中文字符的token数，根据经验，单个汉字的token数大概为1/2~4/3。在使用tiktoken库时，该参数无效
        :return: int 估算的token数。由于不同的API的tokenize方法不同，因此该结果仅供参考
        """
        content = ""
        if history:
            for message in history:
                content += message["content"]
        if message:
            content += message
        if NLGBase.tokenEncoding:
            return len(NLGBase.tokenEncoding.encode(content))
        else:  # 未安装tiktoken库，则只能根据经验进行估算
            # 根据 Moonshot AI的文档(https://platform.moonshot.cn/docs/docs#基本概念介绍)中的经验介绍，
            # token和汉字的比例大约为1:1.5~1:2，而token和英文单词的比例大约为1:1.2，因此我们可以根据这个比例进行估算
            token_per_en_word = 1.2  # 英文单词的token数
            if not token_per_zh_char:
                token_per_zh_char = 1.33  # 取1:1.5
            word_list = re.split("[ ,.?!';:()\[\]{}\t\n]", content)  # 分词
            word_list = [word for word in word_list if word != ""]  # 去除空字符串
            token_num = 0
            for word in word_list:
                if word.isascii():
                    token_num += token_per_en_word
                else:  # 此时为英文与其他语言混合的情况
                    sub_word_list = re.split("[\u4e00-\u9fff]", word)  # 根据中文字符分词
                    zh_char_num = sub_word_list.count("")  # 空字符串的数量即为中文字符的数量
                    token_num += zh_char_num * token_per_zh_char
                    token_num += (len(sub_word_list) - zh_char_num) * token_per_en_word  # 英文单词的数量
            return int(token_num)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class ChatGPT(NLGBase):
    """
    通过API调用ChatGPT进行问答

    API文档参考：https://chatanywhere.apifox.cn/
    """

    def __init__(self, OpenAI_config: dict, prompt: str = None, answer_container=None, condense_question_container=None):
        super().__init__(NLGEnum.ChatGPT, OpenAI_config.get("gpt_model", "gpt-3.5-turbo"), prompt)
        self.api_key = OpenAI_config.get("api_key", None)
        self.temperature = OpenAI_config.get("temperature", 0)
        self.request_timeout = OpenAI_config.get("request_timeout", 30)
        self.streaming = OpenAI_config.get("streaming", False)
        self.answer_container = answer_container
        self.condense_question_container = condense_question_container
        self.chunk_size = OpenAI_config.get("chunk_size", 10000)
        self.chunk_overlap = OpenAI_config.get("chunk_overlap", 0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.max_tokens = OpenAI_config.get("max_tokens", 20000)
        
        if not self.api_key:
            raise ValueError("OpenAI api_key is not set! Please check your 'config.json' file.")
        
        self.chatgpt = ChatOpenAI(
                # max_tokens = self.max_tokens,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                request_timeout=self.request_timeout,
                model=self.model,  # Model name is needed for openai.com only
                streaming=self.streaming,
                callbacks=[StreamHandler(answer_container)] if self.streaming else []
            ) # type: ignore
        
        if self.streaming:
            self.condense_question_llm = ChatOpenAI(
                temperature=self.temperature,
                openai_api_key=self.api_key,
                request_timeout=self.request_timeout,
                model=self.model,  # Model name is needed for openai.com only
                streaming=True,
                callbacks=[StreamHandler(self.condense_question_container, "🤔...")]
            ) # type: ignore
        else:
            self.condense_question_llm = self.chatgpt

        # embeddings = OpenAIEmbeddings(
        #         deployment='"text-embedding-ada-002"',
        #         chunk_size=1
        #         ) # type: ignore
        self.embeddings = HuggingFaceEmbeddings(
            model_name=(
                'sentence-transformers/'
                'multi-qa-MiniLM-L6-cos-v1'
            )
        )

        self.checkConnection()

    

    def init_chatchain(self, chain_type : str = "stuff") -> None:
        # stuff chain_type seems working better than others
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""评价标准:{prompt}
            Chat History:
            {chat_history}

            Follow Up Input:
            {question}

            Standalone Question:"""
            )            
        self.chatchain = ConversationalRetrievalChain.from_llm(llm=self.chatgpt, 
                                                retriever=self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k":4}),
                                                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                condense_question_llm=self.condense_question_llm,
                                                chain_type=chain_type,
                                                return_source_documents=True,
                                                verbose=True)
                                                # combine_docs_chain_kwargs=dict(return_map_steps=False))

    def init_vector_db_from_documents(self, file_list: List[str]=[]):
        docs = []
        for file in file_list:
            print(f"Loading file: {file}")
            ext_name = os.path.splitext(file)[-1]
            # print(ext_name)

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

            doc = loader.load_and_split(self.text_splitter)            
            docs.extend(doc)
            print("Processed document: " + file)

        print("Generating embeddings and ingesting to vector db.")
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        print("Vector db initialized.")
        return {'message':'Vector db initialized.'}

        # print("Vector db initialized.")
        # FAISS.save_local(self.vector_db, 'path', 'index_name')
        # print("Vector db saved to local")

    


    def continuedQuery(self, message: str, history: list[list[str, str]], prompt: str = None):
        
        session_history = []
        for chat in history:
            q = "" if chat[0] == None else chat[0]
            a = "" if chat[1] == None else chat[1]
            # remove details for reference to reduce token
            a = re.sub(r"<details>.*</details>", "", a)
            session_history.append((q, a))       
        response = self.get_chatgpt_result(messages=message, chat_history=session_history, prompt=prompt)
        return response




    def get_chatgpt_result(self, prompt,messages, chat_history):
        result = self.chatchain({
                "prompt": prompt, 
                "question": messages,
                "chat_history": chat_history
        },
        return_only_outputs=True)
        return result['answer']


    def checkConnection(self):
        """
        检查与OpenAI的连接状态(通过一次简单的问答，以测试API可用性)
        :return: bool 是否连接成功
        """
        try:
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [Message(role="user", content="Say this is a test!")]
                },
            )
            if response.status_code != 200:
                raise ConnectionError(f"Connect to {self.model} failed, please check your network and API status.")
            else:
                print(f"Connected to {self.model} successfully.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Connection to {self.model} timed out, please check your network status.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Connect to {self.model} failed, please check your network and API status.")
        except ConnectionError as e:
            raise e
        finally:
            print("OpenAI connection check finished.")