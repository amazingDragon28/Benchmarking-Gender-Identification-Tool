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
    # art = articles.find(filter={'_id': {"$gt": ObjectId('622644e52bbb3f98147f7cde')}})
    # print(articles.count_documents({'_id': {"$gt": ObjectId('622645332bbb3f9814881742')}}))
    # print(art[0]['authors_parsed'][:3])
    # art = articles.find({'authors_parsed': [['Mahdieh', 'Mostafa', '']]})
    # for ar in art:
        # print(ar['_id'])
        # print(ar['authors_parsed'])
    # print(art[0])
    # print(articles.estimated_document_count())
    
    # print(articles.find_one())
    # print(articles.count_documents({'authors_parsed': [['Ghaffari', 'Maani', '']]}))
    # cursor = articles.find({})

    # client = MongoClient()
    # # art = client.arxiv.articles.find_one()
    # articles = client.arxiv.articles
    cursor = articles.find(filter={'_id': {"$gt": ObjectId('622645bf2bbb3f9814955c53')}},projection={'_id': True,'id': True, 'authors': True, 'authors_parsed': True})
    # cursor = articles.find({}, projection={'_id': True,'id': True, 'authors': True, 'authors_parsed': True})
    authors = pd.DataFrame()
    # i = 0
    for art in cursor:
        temp = pd.DataFrame(art['authors_parsed'])
        temp.insert(0, 'paper id', art['id'])
        authors = pd.concat([authors, temp], ignore_index=True)

    authors = authors.rename({0: 'last_name', 1: 'first_name'}, axis=1)
    authors = authors.iloc[:,:3]
    authors.to_csv("scripts/author_2.csv", mode = 'a', index=False, header=False) # remember to delete scripts when final code
    print(authors)

# df = pd.read_csv('author_1.csv')
# df1 = pd.read_csv('author_2.csv')
# df1 = df1.drop(columns=['Unnamed: 0', '3','4','5'])
# df1.columns = ['last name', 'first name', 'other']
# df = pd.concat([df, df1], ignore_index=True)
# print(df.tail())
# print(df.shape)
# df.to_csv('author_1.csv', index=False)
