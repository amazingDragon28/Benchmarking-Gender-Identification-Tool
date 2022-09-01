import os
from pathlib import Path

import pandas as pd
import requests

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
NAMSOR_KEY = '3443475a9c6cdc01197493f47d70b21c'


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
        self.output_file = output_file

    def main(self):
        merged_data = self._query_api_and_merge()
        if Path(self.output_file).exists():
            merged_data = pd.read_csv(self.output_file)
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
        merged_data['api_gender'] = merged_data['api_gender'].fillna('unknown')
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


def run_name_sor(test_data, output_file):
    runner = NameSorRunner(NAMSOR_URL, test_data, output_file)
    runner.main()
