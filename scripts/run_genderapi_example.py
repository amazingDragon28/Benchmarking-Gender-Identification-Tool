import os
from pathlib import Path
import pandas as pd
import requests

GENDER_API_URL = "https://gender-api.com/v2/gender"
GENDER_API_KEY = 'NNnBHUjAS2g3bEaQj5TLR3NDqcScm8hfAVXc'

class GenderApiRunner:
    """
    Query Gender API to guess gender from a person's full name.
    The output file contains the predicted gender, the number of samples
    used for the API to find in the database which match the request, and
    a value between 0 and 1 to determine the reliability of the database.
    
    Require an API key to process request. Please update the API key by changing
    the value of the global variable `GENDER_API_KEY`.
    """
    def __init__(self, url, test_data, output_file):
        self.test_data = test_data
        self.output_file = output_file
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
            output = pd.read_csv(self.output_file)
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

def run_gender_api(test_data, output_file):
    runner = GenderApiRunner(GENDER_API_URL, test_data, output_file)
    runner.main()


