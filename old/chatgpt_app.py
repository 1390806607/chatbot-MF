"""本文件为整个项目的主文件，并使用gradio搭建界面"""
# https://github.com/GaiZhenbiao/ChuanhuChatGPT
from openai import RateLimitError,APIConnectionError
import subprocess
import traceback
from fastapi import FastAPI, Depends, Request
import gradio as gr
from modules import utils
from modules.NLG import ChatGPT,NLGEnum
import uvicorn
import os
from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader, CSVLoader, MWDumpLoader)
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)
from modules.utils import  get_history_names, get_first_history_name
app = FastAPI()
Configs = utils.Configs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chatgpt_service = ChatGPT(Configs["Chatgpt"])


with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot Client", css="./assets/css/GenshinStyle.css",
               js="./assets/js/GenshinStyle.js") as chat_demo:
    with gr.Row(elem_id="baseContainer"):
        with gr.Group():
            with gr.Row(elem_id="history-body"):
                with gr.Column(scale=6, elem_id="history-select-wrap"):
                    historySelectList = gr.Radio(
                        label="从列表中加载对话",
                        choices=get_history_names(user_name='system'),
                        value=get_first_history_name(user_name='system'),
                        # multiselect=False,
                        container=False,
                        elem_id="history-select-dropdown"
                    )
                with gr.Row(visible=False):
                    with gr.Column(min_width=42, scale=1):
                        historyDeleteBtn = gr.Button(
                            "🗑️", elem_id="gr-history-delete-btn")
                    with gr.Column(min_width=42, scale=1):
                        historyDownloadBtn = gr.Button(
                            "⏬", elem_id="gr-history-download-btn")
                    with gr.Column(min_width=42, scale=1):
                        historyMarkdownDownloadBtn = gr.Button(
                            "⤵️", elem_id="gr-history-mardown-download-btn")
        with gr.Column(min_width=280, elem_id="sideBar"):
            # NLGENUM 是大语言模型列表chatgpt  llama等等 [i.name for i in NLGEnum]
            chatgpt_switch = gr.Dropdown(
                [
                    'gpt-3.5-turbo',
                    'gpt-3.5-turbo-16k',
                    'gpt-4',
                    'gpt-4-0125-preview',
                    'gpt-4-turbo-preview',
                    'gpt-4-1106-preview',
                    'gpt-4-vision-preview',
                    'gpt-4-1106-vision-preview',
                    'moonshot-v1-8k',
                    'moonshot-v1-32k',
                    'moonshot-v1-128k'
                ],
                value=Configs['Chatgpt']['gpt_model'], interactive=True,
                label="选择chatgpt模型", 
                elem_id="chatpgtSwitch"
            )
            prompt = gr.Textbox(                
                min_width=80, 
                elem_id="prompt"
                )

            upfile= gr.Files(label="上传文件,支持pdf docx txt等格式", min_width=80, elem_id="upfile")
        with gr.Column(scale=5, elem_id="chatPanel"):
            bot_component = gr.Chatbot(
                min_width=100,
                label='聊天框', 
                avatar_images=utils.getAvatars(), 
                elem_id="chatbot",
                show_label=False,
                show_share_button=False,
                sanitize_html=False,
            )
            with gr.Row(elem_id="inputPanel"):
                text_input = gr.Textbox(placeholder="点击输入", show_label=False, scale=4, elem_id="textInput")
                submit_button = gr.Button(value="发送", size="sm", min_width=80, elem_id="submitButton")
                clear_button = gr.Button(value="清除", size="sm", min_width=80, elem_id="cleanButton")
        
        def read_data(files):
            docs = []
            for file in files:
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

                doc = loader.load_and_split(text_splitter)            
                docs.extend(doc)
            return doc



        def autoChat(files: list, prompt: str, message: str, chat_history: list) -> tuple[str, list[list[str, str]]]:
            """
            自动根据当前前端信息，选择聊天方式进行聊天


            :param message: str 用户输入的消息
            :param chat_history: [[str, str]...] 分别为用户输入和机器人回复(先前的)
            :return: tuple[str, list[list[str, str]]] 空字符串(用以清空输入框), 更新的消息记录
            """
            if not message:
                return "", chat_history,[]
            if files:
                files_data = str(read_data(files))
            else:
                files_data = ''
            if not prompt:
                prompt = ''
            try:
                bot_message = chatgpt_service.continuedQuery(files_data, prompt, message, chat_history)
            
            except RateLimitError as e:
                gr.Warning(e.body['message'])
            except APIConnectionError as e:
                gr.Warning('Api连接错误')
            except Exception as e:
                gr.Warning(e)
            return "", chat_history, files
        
        def switchchatgpt(select_service_name: str):
            """
            切换chatgpt模型
            :param select_service_name: str chatgpt模型名称
            :return: str chatgpt模型名称
            """
            global chatgpt_service, chatgpt_switch
            current_service_name = chatgpt_service.model  # 当前的chatgpt模型名称
            if select_service_name == current_service_name:
                return current_service_name
            else:  # 尝试切换模型
                try:
                    if select_service_name in [
                        'gpt-3.5-turbo',
                        'gpt-3.5-turbo-16k',
                        'gpt-4',
                        'gpt-4-0125-preview',
                        'gpt-4-turbo-preview',
                        'gpt-4-1106-preview',
                        'gpt-4-vision-preview',
                        'gpt-4-1106-vision-preview',
                        'moonshot-v1-8k',
                        'moonshot-v1-32k',
                        'moonshot-v1-128k'
                    ]: 
                        Configs["Chatgpt"]['gpt_model'] = select_service_name
                        temp_service = ChatGPT(utils.Configs["Chatgpt"])

                    else:  # 未知的模型选择，不执行切换
                        gr.Warning(f"未知的chatgpt模型，将不进行切换，当前：{current_service_name}")
                        return current_service_name
                    chatgpt_service = temp_service
                    gr.Info(f"模型切换成功，当前：{chatgpt_service.model}")
                    return chatgpt_service.model
                except Exception:
                    traceback.print_exc()
                    gr.Warning("模型切换失败，请检查网络连接或模型配置")
                    return current_service_name

        def load_chat_history(*args):
            return chatgpt_service.load_chat_history(*args)
        load_history_from_file_args = dict(
            fn=load_chat_history,
            inputs=[historySelectList],
            outputs=[bot_component],
        )

        # 聊天历史数据加载
        historySelectList.input(**load_history_from_file_args)


        # 按钮绑定事件
        clear_button.click(
            fn=lambda message, chat_history, file: ("", [], []),
            inputs=[text_input, bot_component, upfile],
            outputs=[text_input, bot_component, upfile]
        )

        submit_button.click(autoChat, [upfile,prompt,text_input, bot_component], [text_input, bot_component, upfile])
        text_input.submit(autoChat, [upfile,prompt,text_input, bot_component], [text_input, bot_component, upfile])

        # 切换模型
        chatgpt_switch.change(switchchatgpt, [chatgpt_switch], [chatgpt_switch,])
 
app = gr.mount_gradio_app(app, chat_demo, path="/chatbot")
if __name__ == "__main__":
    uvicorn.run(app)