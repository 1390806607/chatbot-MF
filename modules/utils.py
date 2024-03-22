"""本文件中声明了一些常用的函数与全局变量，供其他模块使用。"""
from urllib.parse import urljoin
from datetime import datetime
from json import load, dump
from enum import Enum
import os
import atexit
import uuid
from time import mktime
from typing import TypedDict, Literal
from wsgiref.handlers import format_date_time

import requests

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