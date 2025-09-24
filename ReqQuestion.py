import requests

url = "http://127.0.0.1:5000/ask"
data = {"question": "Quels sont les droits du locataire au Maroc ?"}
response = requests.post(url, json=data)

print(response.json())
