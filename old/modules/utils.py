"""本文件中声明了一些常用的函数与全局变量，供其他模块使用。"""
from urllib.parse import urljoin
from json import load, dump
from enum import Enum
import os
from time import mktime
from typing import TypedDict, Literal
from wsgiref.handlers import format_date_time
from glob import glob
import logging
import datetime
import json

HISTORY_DIR = "history"
try:
    with open('config.json') as cfg:
        Configs = load(cfg)
except FileNotFoundError:
    raise FileNotFoundError(
        "Config file not found! Please make sure you have a 'config.json' file in {}.".format(os.getcwd())
    )


class Message(TypedDict):
    """按照OpenAI的API格式定义的消息类型，可用于检查消息格式是否正确。"""
    role: Literal["user", "assistant", "system"]
    content: str

class NLGEnum(Enum):
    """聊天机器人类型枚举"""
    ChatGPT = 0  # OpenAI ChatGPT
    ChatGLM = 1  # 自行部署的ChatGLM
    ERNIE_Bot = 2  # 百度 文心一言
    Qwen = 3  # 阿里 通义千问
    Gemini = 4  # 谷歌 Gemini
    Spark = 5  # 讯飞 星火大模型
    Waltz = 6  # 自部署 Waltz



def getAvatars() -> tuple[str, str]:
    """
    返回用户头像和bot头像的url链接
    """
    return (
        "./assets/pic/user.jpg",
        "./assets/pic/bot.jpg",
    )





def new_auto_history_filename(username):
    latest_file = get_first_history_name(username)
    if latest_file:
        with open(
            os.path.join(HISTORY_DIR, username, latest_file + ".json"),
            "r",
            encoding="utf-8",
        ) as f:
            if len(f.read()) == 0:
                return latest_file
    now = "新对话 " + datetime.datetime.now().strftime("%m-%d %H-%M")
    return f"{now}.json"

def get_file_names_by_type(dir, filetypes=[".json"]):
    files = glob(dir+'/*.json')
    logging.debug(f"获取文件名列表，目录为{dir}，文件类型为{filetypes}")
    return files

def sorted_by_last_modified_time(list_, dir):
    if list_ is not None:
        return sorted(
            list_, key=lambda char: os.path.getctime(char), reverse=True
        )

def get_file_names_by_last_modified_time(dir, filetypes=[".json"]):
    files = get_file_names_by_type(dir, filetypes)
    if files != [""]:
        files = sorted_by_last_modified_time(files, dir)
    logging.debug(f"files are:{files}")
    return files



def get_history_names(user_name=""):
    logging.debug(f"从用户 {user_name} 中获取历史记录文件名列表")
    # if user_name == "" and hide_history_when_not_logged_in:
    if user_name == "" :
        return []
    else:
        history_files = get_file_names_by_last_modified_time(
            os.path.join(HISTORY_DIR, user_name)
        )
        if history_files is not None:
            history_files = [f[: f.rfind(".")] for f in history_files]
            return history_files
        else:
            return []
    
def get_first_history_name(user_name=""):
    history_names = get_history_names(user_name)
    return history_names[0] if history_names else None


def save_file(filename, model, chatbot):
    system_prompt = model.system_prompt
    history = model.history
    user_name = model.user_name
    os.makedirs(os.path.join(HISTORY_DIR, user_name), exist_ok=True)
    if filename is None:
        filename = new_auto_history_filename(user_name)
    if filename.endswith(".md"):
        filename = filename[:-3]
    if not filename.endswith(".json") and not filename.endswith(".md"):
        filename += ".json"
    if filename == ".json":
        raise Exception("文件名不能为空")

    json_s = {
        "system": system_prompt,
        "history": history,
        "chatbot": chatbot,
        "model_name": model.model,
        # "single_turn": model.single_turn,
        "temperature": model.temperature,
        # "top_p": model.top_p,
        # "n_choices": model.n_choices,
        # "stop_sequence": model.stop_sequence,
        # "token_upper_limit": model.token_upper_limit,
        # "max_generation_token": model.max_generation_token,
        # "presence_penalty": model.presence_penalty,
        # "frequency_penalty": model.frequency_penalty,
        # "logit_bias": model.logit_bias,
        # "user_identifier": model.user_identifier,
        # "metadata": model.metadata,
    }
    if not filename == os.path.basename(filename):
        history_file_path = filename
    else:
        history_file_path = os.path.join(HISTORY_DIR, user_name, filename)

    with open(history_file_path, "w", encoding="utf-8") as f:
        json.dump(json_s, f, ensure_ascii=False, indent=4)

    filename = os.path.basename(filename)
    filename_md = filename[:-5] + ".md"
    md_s = f"system: \n- {system_prompt} \n"
    for data in history:
        # md_s += f"\n{data['role']}: \n- {data['content']} \n"
        md_s += f"\n{data[0]}: \n- {data[1]} \n"
    with open(
        os.path.join(HISTORY_DIR, user_name, filename_md), "w", encoding="utf8"
    ) as f:
        f.write(md_s)
    return os.path.join(HISTORY_DIR, user_name, filename)


