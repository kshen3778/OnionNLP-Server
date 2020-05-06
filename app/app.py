from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
from flask_cors import CORS

# Initialize the app and set a secret_key.
app = Flask(__name__)
CORS(app)

# Load the pickled model.

embedder = pickle.load(open('model.pkl', 'rb'))

#get vector(s) closest to centroid of cluster
def getClosestN(cluster, cluster_centers, embeddings, labels):
    vectors = []
    cosine = []
    for i,vector in enumerate(embeddings):
        if labels[i] == cluster:
            score = cosine_similarity([cluster_centers[cluster]], [embeddings[i]])
            cosine.append(score[0][0])
            vectors.append(i)

    sorted_vector_ids = [x for y, x in sorted(zip(cosine, vectors))]
    sorted_vectors = [ embeddings[i] for i in sorted_vector_ids]
    return sorted_vector_ids, sorted_vectors

@app.route('/get_article', methods=['POST'])
def get_article():
    # Handle empty requests.
    if not request.json:
        return jsonify({'error': 'no request received'})
    print(request.json)
    url = request.json["url"]
    language = request.json["language"]
    article = Article(url, language=language)
    article.download()
    article.parse()
    text = article.text
    title  = article.title
    print(title)
    print(text)
    return jsonify({"title": title, "text": text})

@app.route('/api', methods=['POST'])
def api():
    """Handle request and output model score in json format."""
    # Handle empty requests.
    if not request.json:
        return jsonify({'error': 'no request received'})

    print(request.json)
    text = request.json["text"]
    select_n = request.json["select_n"]


    print("Select n sentences from each cluster: ", select_n)

    #preprocess
    text = text.replace("\n", "")
    text = text.replace(".", ".{split_text}")
    text = text.replace("。", "。{split_text}")
    text = text.replace("?", "?{split_text}")
    text = text.replace("？", "？{split_text}")
    text = text.replace("!", "!{split_text}")
    text = text.replace("！", "！{split_text}")
    text = text.replace('."','."{split_text}')
    text = text.split("{split_text}")
    text = [string for string in text if string != ""]

    embeddings = embedder.encode(text)

    s_scores = [0, 0] #start at 2nd index for simplicity
    for n_cluster in range(2, 8):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(embeddings)
        label = kmeans.labels_
        sil_coeff = silhouette_score(embeddings, label, metric='euclidean')
        s_scores.append(sil_coeff)
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    ind = np.argmax(s_scores)
    print("Maximum value when n_clusters=",ind)

    #Calculate kmeans with optimal cluster
    num_clusters = ind
    Kmean = KMeans(n_clusters=num_clusters, random_state=42)
    Kmean.fit(embeddings)

    #You can get the N closest sentences for each centroid and have them printed out in order
    idsinorder = []
    for cluster in range(num_clusters):
        ids, sorted_vectors = getClosestN(cluster,Kmean.cluster_centers_, embeddings, Kmean.labels_)
        nclosest = ids[:select_n]
        idsinorder += nclosest
    idsinorder.sort()

    response = []
    for i in idsinorder:
        response.append(text[i])
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    #app.run(debug=True) #For localhost testing
    app.run(host='0.0.0.0', port=5000, debug=True) #For production
