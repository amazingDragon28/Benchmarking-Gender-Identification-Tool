# from pymongo import MongoClient
import pandas as pd
# from bson.objectid import ObjectId
from pathlib import Path
# from gender_detector import gender_detector as gd
# import gender_guesser.detector as gender
# from genderize import Genderize
import re
from collections import Counter
import requests
# Replace these with your server details
# MONGO_HOST = "127.0.0.1"
# MONGO_PORT = "27017"
# MONGO_DB = "arxiv"
# MONGO_USER = "mongo_user"
# MONGO_PASS = "monkey!eats!banana"

GENDER_API_KEY = 'NNnBHUjAS2g3bEaQj5TLR3NDqcScm8hfAVXc'
GENDER_API_URL = "https://gender-api.com/v2/gender"

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
NAMSOR_KEY = 'd6604db5c5a3fe8760a9bfefd0038132'

OUTPUT_FILE = Path(__file__).parent.joinpath("../output")

class GenderApiRunner:
    def __init__(self, url, test_data, output_file):
        self.test_data = test_data
        self.output_file = str(output_file)+"/gender_api_arXive.csv"
        self._URL = url
        self._HEADERS = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer" + os.environ["GENDER_TOKEN"]
            "Authorization": "Bearer" + GENDER_API_KEY
        }

    def main(self):
        output = self.query_full_name()
        output = pd.concat([self.test_data, output], axis=1)

        if Path(self.output_file).exists():
            output.to_csv(self.output_file, mode='a', index = False, header=False)
        else:
            output.to_csv(self.output_file, index = False)
        return output

    def query_full_name(self):
        output = pd.DataFrame()

        for name in self.test_data['full_name']:    
            payload = {"full_name": name}
            response = requests.request("POST", self._URL, json=payload, headers=self._HEADERS)
            # print(response.json())
            if 'gender' in response.json():
                output = pd.concat([output, pd.DataFrame({"api_gender": response.json()['gender'], "api_samples": response.json()['details']['samples'], "api_probability": response.json()['probability']}, index=[0])], ignore_index=True)
            else:
                if 'result_found' not in response.json():
                    raise ValueError(response.json()['detail'])
                else:
                    output = pd.concat([output, pd.DataFrame({"api_gender": 'unknown'}, index=[0])], ignore_index=True)
        
        return output

class NameSorRunner:
    """
    If there is previously API output file, will process known data set using namesor API in batches
    of 100 names and then writes output to file. Otherwise will read from the output file.
    Joins these back to the known  dataset by a generated ID and then compares genders between known data and guess.
    Prints the percent concordance

    Requires an environmental variable set: `NAMSOR_KEY` as the API key when registered.
    """

    def __init__(self, url, test_data, output_file):
        self.known_data = test_data.reset_index(drop=True)
        self.known_data['id'] = self.known_data.index
        self._HEADERS = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # "X-API-KEY": os.environ["NAMSOR_KEY"]
            "X-API-KEY": NAMSOR_KEY
        }
        self._URL = url
        self.output_file = str(output_file) + "/namsor_arxiv.csv"

    def main(self):
        merged_data = self._query_api_and_merge()
        if Path(self.output_file).exists():
            # merged_data = pd.read_csv(self.output_file)
            merged_data.to_csv(self.output_file, mode='a', index = False, header=False)
        else:
            merged_data.to_csv(self.output_file, index = False)
            
        return merged_data

    def _query_api_and_merge(self):
        outputs = []
        length = self.known_data.shape[0]
        for lower, upper in zip(range(0, length-99, 100), range(100, length+1, 100)):
            output = self._query_batch(self.known_data, lower, upper)
            outputs.append(output)

        api_out = pd.concat(outputs, ignore_index=True)
        api_out = api_out[['likelyGender', 'script', 'genderScale', 'score', 'probabilityCalibrated']]
        api_out = api_out.rename(columns={'script': 'api_script', 'likelyGender': 'api_gender', 'genderScale': 'api_scale', 'score': 'api_score', 'probabilityCalibrated': 'api_probabilityCalibrated'})
        merged_data = self.known_data.join(api_out, on="id", lsuffix="_api", rsuffix="_known")
        merged_data = merged_data.drop(columns=['id'])
        # merged_data.replace({'female': 'F', 'male': 'M'}, inplace=True)
        # merged_data.to_csv(self.output_file)
        return merged_data

    def _query_batch(self, data, lower, upper):
        subset = data[lower:upper]
        subset = subset.rename(columns={'full_name': 'name'})
        payload_data = [x for x in subset[["name", "id"]].T.to_dict().values()]
        payload = {"personalNames": payload_data}
        response = requests.request("POST", self._URL, json=payload, headers=self._HEADERS)
        return pd.DataFrame(response.json()['personalNames'])


def get_authors_name():
    client = MongoClient()
    articles = client.arxiv.articles
    cursor = articles.find({})
    authors = pd.DataFrame()
    # # i = 0
    # # for art in cursor:
    # #     authors = pd.concat([authors, pd.DataFrame(art['authors_parsed'])], ignore_index=True)
    # for art in articles.find({'_id': {"$gt": ObjectId('622645432bbb3f981489cc2f')}}):
    #     authors = pd.concat([authors, pd.DataFrame(art['authors_parsed'])], ignore_index=True)

    # # authors.columns = ['last name', 'first name', 'other']
    # print(authors.tail())
    # authors.to_csv("author_2.csv")
    return authors

def run_gender_api(test_data):
    runner = GenderApiRunner(GENDER_API_URL, test_data, OUTPUT_FILE)
    return runner.main()

def run_gender_detector(test_data):
    # only identify first name
    detector = gd.GenderDetector()
    output = pd.DataFrame()

    for name in test_data['first_name']:
        try:
            name = re.split("\s|\.\s", name)[0]
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": detector.guess(name)}])], ignore_index=True)
        except:
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": 'unknown'}])], ignore_index=True)

    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv").exists():
        output.to_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv", mode='a', index = False, header=False)
    else:
        output.to_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv", index = False)
    return output

def run_gender_guesser(test_data):
    # case sensitive, only identify firsy name with capital

    dec = gender.Detector(case_sensitive=False)
    output = pd.DataFrame()

    for name in test_data['first_name']:
        output = pd.concat([output, pd.DataFrame({"api_gender": dec.get_gender(name.title()), "api_gender_final": dec.get_gender(name.title())}, index=[0])], ignore_index=True)
    
    output['api_gender'] = output['api_gender_final'].map({'mostly_female':'female', 'mostly_male': 'male', 'andy': 'unknown'}).fillna(output['api_gender_final'])
    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_guesser_arxive.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_arxive.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_arxive.csv", index = False)
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

def run_name_sor(test_data):
    runner = NameSorRunner(NAMSOR_URL, test_data, OUTPUT_FILE)
    return runner.main()

def determine_gender(row):
    count = Counter(list(row))
    if count.most_common()[0][1] == 4:
        return count.most_common()[0][0], 1, 'all'
    
    if (count.most_common()[0][0] == 'unknown') and (count['male'] != count['female']):
        return count.most_common()[1][0], 0.25, [row[row == count.most_common()[1][0]].index[0]]
    elif (count.most_common()[0][0] == 'unknown') and (count['male'] == count['female']):
        return 'binary', 0.25, (row[row == 'male'].index[0], row[row == 'female'].index[0])
    elif (count['male'] == count['female']) and (count['andy'] == 0):
        return 'binary', count['male'] / 4, (list(row[row == 'male'].index.values), list(row[row == 'female'].index.values))
    elif (count['male'] == count['female']) and (count['andy'] != 0):
        return 'binary', (count['male'] + 1) / 4, (row[row == 'male'].index[0], 'gg_gender', row[row == 'female'].index[0], 'gg_gender')
    else:
        # print("test")
        return count.most_common()[0][0], count.most_common()[0][1] / 4, list(row[row == count.most_common()[0][0]].index.values)

def cal_confidence(row):
    tools_map = {
        'ga_gender': 'ga_probability',
        'genderize_gender': 'genderize_probability',
        'gd_gender': 0.87, # highest threshold used: 0.99, lowest threshold: 0.75
        'gg_gender': 0.75  
    }
    if row[1] == 1:
        confidence = 1
    else:
        probability = [genders[tools_map[x]][row.name] if (x == 'ga_gender') or (x == 'genderize_gender') else tools_map[x] for x in row[2]]
        confidence = row[1] * (sum(probability) / len(probability)) 
    return confidence

if __name__ == "__main__":
    
    # authors = get_authors_name()

    # authors = pd.read_csv("author_2.csv")
    # print(authors.shape)

    # # drop rows with NaN first name
    # authors = authors.dropna(subset=['first_name'])
    # print(authors.shape)

    # # drop rows with initials first name
    # ini_index = authors[authors['first_name'].str.match(r'[A-Za-z]\.')].index
    # authors = authors.drop(ini_index)
    # authors = authors.reset_index(drop=True)
    # authors.to_csv("author_without_initial.csv", index=False)
    # print(authors.shape)

    # get the most frequency 10000 names
    # authors_10000 = authors.groupby(['first_name']).size().to_frame('count').reset_index() \
    #          .sort_values('count', ascending=False) \
    #          .drop_duplicates(subset='first_name')

    # authors_10000 = authors_10000[:10000].apply(lambda x: authors[authors['first_name'] == x[0]].iloc[0], axis=1)
    # authors_10000.to_csv('author_10000.csv', index=False)

    # drop duplicate first names
    # authors = authors.drop_duplicates(subset='first_name', keep="first")
    # authors['full_name'] = authors.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis = 1)
    # authors.to_csv("author_unique.csv", index=False)
    # print(authors.shape)

    authors = pd.read_csv("scripts/author_unique.csv")
    # authors = pd.read_csv("scripts/author_10000.csv")
    # authors['full_name'] = authors.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis = 1)
    # authors.to_csv('scripts/author_10000.csv', index=False)
    
    # output_ga = run_gender_api(test)
    # output_gd = run_gender_detector(authors)
    # output_gg = run_gender_guesser(authors)
    # output_genderize = run_genderize(authors)
    # output_namsor = run_name_sor(authors_10000)

    # output_ga = pd.read_csv(str(OUTPUT_FILE)+"/gender_api_arxiv.csv")
    # output_gd = pd.read_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv")
    # output_gg = pd.read_csv(str(OUTPUT_FILE)+"/gender_guesser_arxiv.csv")
    # output_genderize = pd.read_csv(str(OUTPUT_FILE)+"/genderize_arxiv.csv")

    # # df = pd.concat([output_gd.set_index('first name'), output_gg.set_index('first name'), output_genderize.set_index('first name')], axis='columns', keys=['gender detector', 'gender-guesser', 'genderize'])
    # genders = pd.concat([test, output_ga['api_gender'], output_gd['api_gender'], output_gg['api_gender_final'], output_genderize['api_gender'], output_ga['api_probability'], output_genderize['api_probability']], axis='columns')
    # genders.columns = ['last_name', 'first_name', 'other', 'full_name', 'ga_gender', 'gd_gender', 'gg_gender', 'genderize_gender', 'ga_probability', 'genderize_probability']
    # print(genders.head())
    # genders = pd.read_csv('scripts/gender_test.csv')
    # # gender1 = pd.DataFrame()
    # gender1 = genders.loc[:,'ga_gender':'genderize_gender'].apply(determine_gender, axis=1, result_type='expand')
    # genders.insert(4, 'gender', gender1[0])

    # # TODO: remove unknown gender and record the paper id
    # genders = genders.drop(genders[genders['gender'] == 'unknown'].index)
    # genders = genders.reset_index(drop=True)

    # gender1 = gender1.drop(genders[genders['gender'] == 'unknown'].index)
    # gender1 = gender1.reset_index(drop=True)
    # genders.insert(5, 'confidence', gender1.apply(cal_confidence, axis=1))

    # print("hh")
    # print("hh")