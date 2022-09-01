import os
from pathlib import Path
import pandas as pd
import requests

# API_KEY = "843bd95c449fa0a67b2a9873bfa476b3-user1"
API_KEY = "3164de9b685a45b5534e8241e0abf4c2-user1"
NAME_API_URL = "https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey=" + API_KEY


class NameApiRunner:
    """
    Use Name API to guess gender from a full name. The output file contains the predicted gender
    and confidence score.

    Require an API key. You can replace value of the global variable `API_KEY` with your own Name API key.
    """
    def __init__(self, url, test_data, output_file):
        self.test_data = test_data
        self.output_file = output_file
        self._URL = url

    def main(self):
        output = self.query_full_name()
        output = pd.concat([self.test_data, output], axis=1)

        if Path(self.output_file).exists():
            output = pd.read_csv(self.output_file)
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
        
        output['api_gender'] = output['api_gender'].map({'FEMALE':'female','MALE':'male'}).fillna('unknown')
        return output

def run_name_api(test_data, output_file):
    runner = NameApiRunner(NAME_API_URL, test_data, output_file)
    runner.main()
