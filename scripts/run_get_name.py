from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId
from pathlib import Path

# Replace these with your server details
MONGO_HOST = "127.0.0.1"
MONGO_PORT = "27017"
MONGO_DB = "arxiv"
MONGO_USER = "mongo_user"
MONGO_PASS = "monkey!eats!banana"

DATA_FILE = Path(__file__).parent.joinpath("../data")

def get_authors_name():
    """
    Obtained all paper id and author names of arXiv data from MongoDB
    """

    client = MongoClient()
    articles = client.arxiv.articles
    cursor = articles.find({} ,projection={'_id': True,'id': True, 'authors': True, 'authors_parsed': True})
    authors = pd.DataFrame()

    for art in cursor:
        temp = pd.DataFrame(art['authors_parsed'])
        temp.insert(0, 'paper id', art['id'])
        authors = pd.concat([authors, temp], ignore_index=True)

    authors = authors.rename({0: 'last_name', 1: 'first_name'}, axis=1)
    authors = authors.iloc[:,:3]
    
    authors.to_csv(str(DATA_FILE) + "/author.csv", index=False)
    return authors