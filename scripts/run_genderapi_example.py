import os
from pathlib import Path
import pandas as pd
import requests

GENDER_API_URL = "https://gender-api.com/v2/gender"
TEST_FILE = Path(__file__).parent.joinpath("../data/test_names_1000.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output/gender_api_output.csv")

class GenderApiRunner:
    def __init__(self, url, test_file, output_file):
        self.test_data = self.read_data(test_file)
        self.output_file = output_file
        self._URL = url
        self._HEADERS = {
            "Content-Type": "application/json",
            "Authorization": "Bearer" + os.environ["GENDER_TOKEN"]
        }

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
            payload = {"full_name": name}
            response = requests.request("POST", self._URL, json=payload, headers=self._HEADERS)
            if 'gender' in response.json():
                output = pd.concat([output, pd.DataFrame({"name": name, "gender": response.json()['gender']}, index=[0])], ignore_index=True)
            else:
                if 'result_found' not in response.json():
                    raise ValueError(response.json()['detail'])
                else:
                    output = pd.concat([output, pd.DataFrame({"name": name, "gender": 'N'}, index=[0])], ignore_index=True)

        output['gender'] = output['gender'].map({'female':'F','male':'M'})
        return output

if __name__ == "__main__":
    runner = GenderApiRunner(GENDER_API_URL, TEST_FILE, OUTPUT_FILE)
    runner.main()

