from pymongo import MongoClient

# Replace these with your server details
MONGO_HOST = "127.0.0.1"
MONGO_PORT = "27017"
MONGO_DB = "arxiv"
MONGO_USER = "mongo_user"
MONGO_PASS = "monkey!eats!banana"


if __name__ == "__main__":
    client = MongoClient()
    art = client.arxiv.articles.find_one()
    print(art)

