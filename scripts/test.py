import pandas as pd
from bson.objectid import ObjectId
from pymongo import MongoClient
import pymongo

# Replace these with your server details
MONGO_HOST = "127.0.0.1"
MONGO_PORT = "27017"
MONGO_DB = "arxiv"
MONGO_USER = "mongo_user"
MONGO_PASS = "monkey!eats!banana"


if __name__ == "__main__":
    client = MongoClient()
    # art = client.arxiv.articles.find_one()
    articles = client.arxiv.articles
    # art = articles.find({'authors_parsed': [['Aharonov', 'Dov', ''], ['Elias', 'Uri', '']]})
    # print(art[0]['id'])
    # art = articles.find(filter={'_id': {"$lt": ObjectId('622645432bbb3f981489cc2f')}}, sort = [('_id', -1)])
    
    # print(articles.count_documents({'_id': {"$gt": ObjectId('622645432bbb3f981489cc2f')}}))
    # print(art[0]['authors_parsed'][:3])
    art = articles.find({'authors_parsed': [['Mahdieh', 'Mostafa', '']]})
    # for ar in art:
        # print(ar['_id'])
        # print(ar['authors_parsed'])
    # print(art[0])
    print(art[0])
    # print(articles.count_documents({'authors_parsed': [['Ghaffari', 'Maani', '']]}))
    # cursor = articles.find({})

# df = pd.read_csv('author_1.csv')
# df1 = pd.read_csv('author_2.csv')
# df1 = df1.drop(columns=['Unnamed: 0', '3','4','5'])
# df1.columns = ['last name', 'first name', 'other']
# df = pd.concat([df, df1], ignore_index=True)
# print(df.tail())
# print(df.shape)
# df.to_csv('author_1.csv', index=False)
