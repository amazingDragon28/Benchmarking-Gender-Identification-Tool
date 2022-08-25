import os
from pathlib import Path
import pandas as pd
import requests

# API_KEY = "843bd95c449fa0a67b2a9873bfa476b3-user1"
API_KEY = "3164de9b685a45b5534e8241e0abf4c2-user1"
NAME_API_URL = "https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey=" + API_KEY
TEST_FILE = Path(__file__).parent.joinpath("../data/test_names_1000.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output/name_api_output.csv")


class NameApiRunner:
    def __init__(self, url, test_file, output_file):
        self.test_data = self.read_data(test_file)
        self.output_file = output_file
        self._URL = url

    def read_data(self, test_file):
        data = pd.read_csv(test_file)
        return data

    def main(self):
        output = self.query_full_name()
        if Path(self.output_file).exists():
            output.to_csv(self.output_file, mode='a', index = False, header=False)
        else:
            output.to_csv(self.output_file, index = False)

        # Calculate accurate
        diff = self.test_data['Gender'].compare(output['gender'])
        accuracy = (output.shape[0] - diff.shape[0]) / output.shape[0]
        print("The accuracy is {:.2%}".format(accuracy))

    def query_full_name(self):
        output = pd.DataFrame()
        for name in self.test_data['Name']:    
            payload = {
                "inputPerson": {
                    "type": "NaturalInputPerson",
                    "personName":{
                        "nameFields": [ {
                            "string": name.split(" ")[0],
                            "fieldType": "GIVENNAME"
                        }, {
                            "string": name.split(" ")[-1],
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
                    output = pd.concat([output, pd.DataFrame({"name": name, "gender": response.json()['gender']}, index=[0])], ignore_index=True)
                else:
                    output = pd.concat([output, pd.DataFrame({"name": name, "gender": 'N'}, index=[0])], ignore_index=True)
        
        output['gender'] = output['gender'].map({'FEMALE':'F','MALE':'M', "UNKNOWN": 'N'})
        return output

if __name__ == "__main__":
    runner = NameApiRunner(NAME_API_URL, TEST_FILE, OUTPUT_FILE)
    runner.main()

