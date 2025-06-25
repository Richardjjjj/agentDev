import os
import requests

headers = {
    'Authorization': 'Bearer ' + "na",
    'Content-Type': 'application/json',
}

json_data = {
    'input': '北京有哪些景点',
    'model': 'gpt-4',
}

response = requests.post('http://192.168.10.137:8200/v1/embeddings', headers=headers, json=json_data)
print(response.text)