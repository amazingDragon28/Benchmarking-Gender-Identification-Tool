from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId
from pathlib import Path
from gender_detector import gender_detector as gd
import gender_guesser.detector as gender
from genderize import Genderize
import re
# Replace these with your server details
# MONGO_HOST = "127.0.0.1"
# MONGO_PORT = "27017"
# MONGO_DB = "arxiv"
# MONGO_USER = "mongo_user"
# MONGO_PASS = "monkey!eats!banana"

OUTPUT_FILE = Path(__file__).parent.joinpath("../output")

def get_authors_name():
    # client = MongoClient()
    # # art = client.arxiv.articles.find_one()
    # articles = client.arxiv.articles
    # # cursor = articles.find({})
    # authors = pd.DataFrame()
    # # i = 0
    # # for art in cursor:
    # #     authors = pd.concat([authors, pd.DataFrame(art['authors_parsed'])], ignore_index=True)
    # for art in articles.find({'_id': {"$gt": ObjectId('622645432bbb3f981489cc2f')}}):
    #     authors = pd.concat([authors, pd.DataFrame(art['authors_parsed'])], ignore_index=True)

    # # authors.columns = ['last name', 'first name', 'other']
    # print(authors.tail())
    # authors.to_csv("author_2.csv")
    return authors

def run_gender_detector(test_data):
    # only identify first name
    detector = gd.GenderDetector()
    output = pd.DataFrame()

    for name in test_data['first name']:
        try:
            name = re.split("\s|\.\s", name)[0]
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": detector.guess(name)}])], ignore_index=True)
        except:
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": 'unknown'}])], ignore_index=True)

    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv", index = False)
    return output

def run_gender_guesser(test_data):
    # case sensitive, only identify firsy name with capital

    dec = gender.Detector()
    output = pd.DataFrame()

    for name in test_data['first name']:
        name = re.split("\s|\.\s", name)[0]
        output = pd.concat([output, pd.DataFrame({"api_gender": dec.get_gender(name.title()), "api_gender_final": dec.get_gender(name.title())}, index=[0])], ignore_index=True)
    
    output['api_gender_final'] = output['api_gender_final'].map({'mostly_female':'female', 'mostly_male': 'male'}).fillna(output['api_gender_final'])
    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_arxiv.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_arxiv.csv", index = False)
    return output

def run_genderize(test_data):
    # only identify first name

    output = pd.DataFrame(Genderize().get(test_data['first name'].str.split(r"\s|\.\s", expand=True)[0]))

    output = output[['gender', 'count', 'probability']]
    output['gender'] = output['gender'].map({None:'unknown'}).fillna(output['gender'])
    output = output.rename(columns={'gender': 'api_gender', 'count': 'api_count', 'probability': 'api_probability'})

    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/genderize_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/genderize_arxiv.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/genderize_arxiv.csv", index = False)
    return output

if __name__ == "__main__":
    
    # authors = get_authors_name()

    # authors = pd.read_csv("author_1.csv")
    # print(authors.shape)

    # # drop rows with NaN first name
    # authors = authors.dropna(subset=['first name'])
    # print(authors.shape)

    # # drop rows with initials first name
    # ini_index = authors[authors['first name'].str.match(r'[A-Za-z]\.')].index
    # authors = authors.drop(ini_index)
    # authors = authors.reset_index(drop=True)
    # authors.to_csv("author_without_initial.csv", index=False)
    # print(authors.shape)

    # # drop duplicate first names
    # authors = authors.drop_duplicates(subset='first name', keep="first")
    # authors.to_csv("author_unique.csv", index=False)
    # print(authors.shape)

    authors = pd.read_csv("scripts/author_unique.csv")

    test = authors.sample(n = 100)
    test = test.reset_index(drop=True)
    output_gd = run_gender_detector(test)
    output_gg = run_gender_guesser(test)
    output_genderize = run_genderize(test)

    # df = pd.concat([output_gd.set_index('first name'), output_gg.set_index('first name'), output_genderize.set_index('first name')], axis='columns', keys=['gender detector', 'gender-guesser', 'genderize'])

    print("hh")