import os
from openai import OpenAI
 
client = OpenAI(
    api_key="sk-TaTGR1X8xM22btHub6zP2L1rjUTUQzyoagDGyAQujs0hkoZe",
    base_url="https://api.moonshot.cn/v1",
)
 
model_list = client.models.list()
model_data = model_list.data
 
for i, model in enumerate(model_data):
    print(f"model[{i}]:", model.id)