import os
import pandas as pd
import requests
from gender_detector import gender_detector as gd
import gender_guesser.detector as gender
from genderize import Genderize
import numpy as np

from pathlib import Path

GENDER_API_KEY = ''
GENDER_API_URL = "https://gender-api.com/v2/gender"

NAME_API_KEY = ''
NAME_API_URL = "https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey=" + NAME_API_KEY

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
NAMSOR_KEY = ''

TEST_FILE = Path(__file__).parent.joinpath("../data/pubmed.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output")

class GenderApiRunner:
    def __init__(self, url, test_data, output_file):
        self.test_data = test_data
        self.output_file = str(output_file)+"/gender_api_pubmed.csv"
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


class NameApiRunner:
    def __init__(self, url, test_data, output_file):
        self.test_data = test_data
        self.output_file = str(output_file)+"/name_api_pubmed.csv"
        self._URL = url

    def main(self):
        output = self.query_full_name()
        output = pd.concat([test_data, output], axis=1)

        if Path(self.output_file).exists():
            output.to_csv(self.output_file, mode='a', index = False, header=False)
        else:
            output.to_csv(self.output_file, index = False)

    def query_full_name(self):
        output = pd.DataFrame()
        for index, name in self.test_data.iterrows():    
            payload = {
                "inputPerson": {
                    "type": "NaturalInputPerson",
                    "personName":{
                        "nameFields": [ {
                            "string": name['first_name'],
                            "fieldType": "GIVENNAME"
                        }, {
                            "string": name['last_name'],
                            "fieldType": "SURNAME"
                        }]
                    },
                    "gender": "UNKNOWN"
                }
            }
            response = requests.request("POST", self._URL, json=payload)
            # print(response.json())
            if 'httpStatusCode' in response.json():
                raise requests.exceptions.HTTPError(response.json()['message'])
            else:
                if 'gender' in response.json():
                    output = pd.concat([output, pd.DataFrame({"api_gender": response.json()['gender'], 'api_confidence': response.json()['confidence']}, index=[0])], ignore_index=True)
                else:
                    output = pd.concat([output, pd.DataFrame({"api_gender": 'unknown', 'api_confidence': 0}, index=[0])], ignore_index=True)
        
        output['api_gender'] = output['api_gender'].map({'FEMALE':'female','MALE':'male', "UNKNOWN": 'unknown'}).fillna(output['api_gender'])
        output.to_csv("output_test.csv", index=False)
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
        self.known_data = test_data
        self.known_data['id'] = self.known_data.index
        self._HEADERS = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # "X-API-KEY": os.environ["NAMSOR_KEY"]
            "X-API-KEY": NAMSOR_KEY
        }
        self._URL = url
        self.output_file = str(output_file) + "/namsor_pubmed.csv"

    def main(self):
        if Path(self.output_file).exists():
            merged_data = pd.read_csv(self.output_file)
        else:
            merged_data = self._query_api_and_merge()

    def _query_api_and_merge(self):
        outputs = []
        for lower, upper in zip(range(0, 1901, 100), range(100, 2001, 100)):
            output = self._query_batch(self.known_data, lower, upper)
            outputs.append(output)

        api_out = pd.concat(outputs, ignore_index=True)
        api_out = api_out[['likelyGender', 'script', 'genderScale', 'score', 'probabilityCalibrated']]
        api_out = api_out.rename(columns={'script': 'api_script', 'likelyGender': 'api_gender', 'genderScale': 'api_scale', 'score': 'api_score', 'probabilityCalibrated': 'api_probabilityCalibrated'})
        merged_data = self.known_data.join(api_out, on="id", lsuffix="_api", rsuffix="_known")
        merged_data = merged_data.drop(columns=['id'])
        # merged_data.replace({'female': 'F', 'male': 'M'}, inplace=True)
        merged_data.to_csv(self.output_file)
        return merged_data

    def _query_batch(self, data, lower, upper):
        subset = data[lower:upper]
        subset = subset.rename(columns={'full_name': 'name'})
        payload_data = [x for x in subset[["name", "id"]].T.to_dict().values()]
        payload = {"personalNames": payload_data}
        response = requests.request("POST", self._URL, json=payload, headers=self._HEADERS)
        return pd.DataFrame(response.json()['personalNames'])


def run_gender_api(test_data):
    runner = GenderApiRunner(GENDER_API_URL, test_data, OUTPUT_FILE)
    runner.main()

def run_gender_detector(test_data):
    # only identify first name
    detector = gd.GenderDetector()
    output = pd.DataFrame()

    for name in test_data['first_name']:
        try:
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": detector.guess(name)}])], ignore_index=True)
        except:
            output = pd.concat([output, pd.DataFrame.from_records([{"api_gender": 'unknown'}])], ignore_index=True)

    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv", index = False)

def run_gender_guesser(test_data):
    # case sensitive, only identify firsy name with capital

    dec = gender.Detector(case_sensitive=False)
    output = pd.DataFrame()

    for name in test_data['first_name']:
        output = pd.concat([output, pd.DataFrame({"api_gender": dec.get_gender(name.title()), "api_gender_final": dec.get_gender(name.title())}, index=[0])], ignore_index=True)
    
    output['api_gender_final'] = output['api_gender_final'].map({'mostly_female':'female', 'mostly_male': 'male'}).fillna(output['api_gender_final'])
    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv", index = False)


def run_genderize(test_data):
    # only identify first name

    output = pd.DataFrame(Genderize().get(test_data['first_name']))

    output = output[['gender', 'count', 'probability']]
    output['gender'] = output['gender'].map({None:'unknown'}).fillna(output['gender'])
    output = output.rename(columns={'gender': 'api_gender', 'count': 'api_count', 'probability': 'api_probability'})

    output = pd.concat([test_data, output], axis=1)

    if Path(str(OUTPUT_FILE)+"/genderize_pubmed.csv").exists():
            output.to_csv(str(OUTPUT_FILE)+"/genderize_pubmed.csv", mode='a', index = False, header=False)
    else:
            output.to_csv(str(OUTPUT_FILE)+"/genderize_pubmed.csv", index = False)


def run_name_api(test_data):
    runner = NameApiRunner(NAME_API_URL, test_data, OUTPUT_FILE)
    runner.main()

def run_name_sor(test_data):
    runner = NameSorRunner(NAMSOR_URL, test_data, OUTPUT_FILE)
    runner.main()

def cal_metrics(data_file):
    api_data = pd.read_csv(data_file)
    mm_df = np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'male'))
    mm = mm_df[0].size

    mf_df = np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'female'))
    mf = mf_df[0].size

    mu_df= np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'unknown'))
    mu = mu_df[0].size

    fm_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'male'))
    fm = fm_df[0].size

    ff_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'female'))
    ff = ff_df[0].size

    fu_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'unknown'))
    fu = fu_df[0].size

    # unknown = np.where((api_data['gender'] == 'unknown'))
    errorCoded = (fm + mf + mu + fu) / (mm + fm + mf + ff + mu + fu)
    errorCodedWithoutNA = (fm + mf) / (mm + fm + mf + ff)
    naCoded = (mu + fu) / (mm + fm + mf + ff + mu + fu)
    errorGenderBias = (mf - fm) / (mm + fm + mf + ff)

    # print("errorCoded: %f" % errorCoded)
    # print("errorCodedWithoutNA: %f" % errorCodedWithoutNA)
    # print("naCoded: %f" % naCoded)
    # print("errorGenderBias: %f" % errorGenderBias)

    return [errorCoded, errorCodedWithoutNA, naCoded, errorGenderBias], [mm, mf, mu, fm, ff, fu]

    

if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)
    test_data = test_data.iloc[:, 0:5]
    test_data['gender'] = test_data['gender'].map({'f': 'female', 'm': 'male', 'u': 'unknown'})
    
    run_gender_api(test_data)
    run_gender_detector(test_data)
    run_gender_guesser(test_data)
    run_genderize(test_data)
    run_name_api(test_data)
    run_name_sor(test_data)

    # print("gender api")
    gender_api_metrics, gender_api_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_api_pubmed.csv")

    # print("gender detector")
    gender_detector_metrics, gender_detector_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv")

    # print("gender guesser")
    gender_guesser_metrics, gender_guesser_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv")

    # print("genderize")
    genderize_metrics, genderize_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/genderize_pubmed.csv")

    # print("name api")
    name_api_metrics, name_api_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/name_api_pubmed.csv")

    # print("NamSor")
    namsor_metrics, namsor_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/namsor_pubmed.csv")

    list_of_metrics = [gender_api_metrics, gender_detector_metrics, gender_guesser_metrics, genderize_metrics, name_api_metrics, namsor_metrics]
    metrics = pd.DataFrame(list_of_metrics, index=['Gender API', 'Gender Detector', 'Gender Guesser', 'Genderize', 'Name API', 'NamSor'], 
    columns=['errorCoded', 'errorCodedWithouNA', 'naCoded', 'errorGenderBias'])

    print(metrics)
    metrics.to_csv(str(OUTPUT_FILE) + "/tools_performance_pubmed.csv")

    iterables = [['Gender API', 'gender-detector', 'gender-guesser', 'genderize.io', 'Name API', 'NamSor'], ['male', 'female']]
    index = pd.MultiIndex.from_product(iterables) 
    list_of_confusion_matrix = [gender_api_confusion_matrix[:3], gender_api_confusion_matrix[3:], gender_detector_confusion_matrix[:3], gender_detector_confusion_matrix[3:],
                                gender_guesser_confusion_matrix[:3], gender_guesser_confusion_matrix[3:], genderize_confusion_matrix[:3], genderize_confusion_matrix[3:], 
                                name_api_confusion_matrix[:3], name_api_confusion_matrix[3:], namsor_confusion_matrix[:3], namsor_confusion_matrix[3:]]
    confusion_matrix = pd.DataFrame(list_of_confusion_matrix, index=index, 
    columns=['Predicted as male', 'Predicted as female', 'Predicted as unknown'])

    print(confusion_matrix)  
    confusion_matrix.to_csv(str(OUTPUT_FILE) + "/tools_confusion_matrix_pubmed.csv") 
