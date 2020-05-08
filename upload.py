import pickle
import json
import requests
from sentence_transformers import SentenceTransformer
import ast
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from bs4 import BeautifulSoup
from newspaper import Article


# embedder = SentenceTransformer('distiluse-base-multilingual-cased')
# filename = 'model.pkl'
# pickle.dump(embedder, open(filename, 'wb'))


#main_url = "https://www.bbc.com/news/world-europe-52521426"
main_url = "https://www.chinadialogue.net/article/show/single/ch/11988-Shelving-of-huge-BRI-coal-plant-highlights-overcapacity-risk-in-Pakistan-and-Bangladesh-"
language = "zh"
article = Article(main_url, language=language)
article.download()
article.parse()
text = article.text

#Test web scraping
url = "https://onion-backend-ef2kdcjinq-ue.a.run.app/get_article"
data = {'url': "", "language": ""}
data["url"] = "https://www.chinadialogue.net/article/show/single/ch/11988-Shelving-of-huge-BRI-coal-plant-highlights-overcapacity-risk-in-Pakistan-and-Bangladesh-"
data["language"] = "zh"
headers = {'Content-type': 'application/json'}
r = requests.post(url, data=json.dumps(data), headers=headers)
response_obj = r.content.decode("utf-8")
print(response_obj)
print(ast.literal_eval(response_obj))


# #Test summarization
# url = "http://0.0.0.0:5000/api"
# data = {'text': ""}
# data["text"] = text
# data["select_n"] = 5
# headers = {'Content-type': 'application/json'}
# r = requests.get(url, data=json.dumps(data), headers=headers)
# response_obj = r.content.decode("utf-8")
# #print(response_obj)
# print(ast.literal_eval(response_obj))
# for i in ast.literal_eval(response_obj):
#     print(i)



# cred = credentials.Certificate('service_key.json')
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://onionnlp.firebaseio.com'
# })
# ref = db.reference('/')
# print(ref.get())
