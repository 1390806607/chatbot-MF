"""本文件为整个项目的主文件，并使用gradio搭建界面"""
# https://github.com/GaiZhenbiao/ChuanhuChatGPT
import subprocess
import traceback
from fastapi import FastAPI, Depends, Request
import gradio as gr
from modules import utils
from modules.NLG import ChatGPT,NLGEnum
import uvicorn
import os

Configs = utils.Configs
chatgpt_service = ChatGPT(Configs["Chatgpt"])
app = FastAPI()





with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot Client", css="./assets/css/GenshinStyle.css",
               js="./assets/js/GenshinStyle.js") as chat_demo:
    with gr.Row(elem_id="baseContainer"):
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
                    'gpt-4-1106-vision-preview'
                ],
                value=chatgpt_service.model, interactive=True,
                label="选择chatgpt模型", 
                elem_id="chatpgtSwitch"
            )
            prompt = gr.Textbox(                
                min_width=80, 
                elem_id="prompt"
                )

            upfile= gr.Files(label="上传文件,支持pdf docx txt等格式", min_width=80, elem_id="upfile")
            submit_files_button = gr.Button(value="确定prompt", size="sm", min_width=80, elem_id="submit_files_button")

        with gr.Column(scale=5, elem_id="chatPanel"):
            bot_component = gr.Chatbot(min_width=100,label=chatgpt_service.model, avatar_images=utils.getAvatars(), elem_id="chatbot")
            with gr.Row(elem_id="inputPanel"):
                text_input = gr.Textbox(placeholder="点击输入", show_label=False, scale=4, elem_id="textInput")
                submit_button = gr.Button(value="发送", size="sm", min_width=80, elem_id="submitButton")
                clear_button = gr.Button(value="清除", size="sm", min_width=80, elem_id="cleanButton")
        

        def ingest_docs_to_vector_store(files: list):
            if files:
                gr.Info('请等待一会, 需要评估的内容正在上传')
                result = chatgpt_service.init_vector_db_from_documents(file_list=files)
                gr.Info(result['message'])
                chatgpt_service.init_chatchain()
            else:
                gr.Error('请上传文档')

        def autoChat(prompt: str, message: str, chat_history: list) -> tuple[str, list[list[str, str]]]:
            """
            自动根据当前前端信息，选择聊天方式进行聊天


            :param message: str 用户输入的消息
            :param chat_history: [[str, str]...] 分别为用户输入和机器人回复(先前的)
            :return: tuple[str, list[list[str, str]]] 空字符串(用以清空输入框), 更新的消息记录
            """
            if not message:
                return "", chat_history
            if prompt:
                message = '评价标准:' + prompt + '\n' + message
            bot_message = chatgpt_service.continuedQuery(message, chat_history, prompt=prompt)
            chat_history.append((message, bot_message))
            return "", chat_history
        
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
                    if select_service_name in ['gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-4']: 
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


        # 按钮绑定事件
        clear_button.click(
            fn=lambda message, chat_history, file: ("", [], []),
            inputs=[text_input, bot_component, upfile],
            outputs=[text_input, bot_component, upfile]
        )
        submit_files_button.click(ingest_docs_to_vector_store, [upfile])
        submit_button.click(autoChat, [prompt,text_input, bot_component], [text_input, bot_component])
        # text_input.submit(textChat, [upfile,text_input, bot_component], [text_input, bot_component])

        # 切换模型
        chatgpt_switch.change(switchchatgpt, [chatgpt_switch], [chatgpt_switch])
 
app = gr.mount_gradio_app(app, chat_demo, path="/chatbot")
if __name__ == "__main__":
    uvicorn.run(app, host='192.168.2.35')
    # uvicorn.run(app)