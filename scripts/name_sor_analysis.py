import os
from pathlib import Path

import pandas as pd
import requests

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
OUTPUT_FILE = Path(__file__).parent.parent / "output" / "name_sor_out.csv"
KNOWN_DATA_FILE = Path(__file__).parent.parent / "data" / "test_names_1000.csv"


class NameSorRunner:
    """
    If there is previously API output file, will process known data set using namesor API in batches
    of 100 names and then writes output to file. Otherwise will read from the output file.
    Joins these back to the known  dataset by a generated ID and then compares genders between known data and guess.
    Prints the percent concordance

    Requires an environmental variable set: `NAMSOR_KEY` as the API key when registered.
    """

    def __init__(self, url, output_file, known_data_file):
        self.known_data = self._read_test_data(known_data_file)
        self._HEADERS = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": os.environ["NAMSOR_KEY"]
        }
        self._URL = url
        self.output_file = output_file

    @staticmethod
    def _read_test_data(known_data_file):
        data = pd.read_csv(known_data_file)
        data['id'] = data.index
        data = data.rename(columns={"Name": "name"})
        return data

    def main(self):
        if Path(self.output_file).exists():
            merged_data = pd.read_csv(self.output_file)
        else:
            merged_data = self._query_api_and_merge()
        concordance = len(merged_data[merged_data.Gender == merged_data.likelyGender]) / len(merged_data)
        print(f"Percent concordance: {concordance * 100}")

    def _query_api_and_merge(self):
        outputs = []
        for lower, upper in zip(range(0, 1901, 100), range(100, 2001, 100)):
            output = self._query_batch(self.known_data, lower, upper)
            outputs.append(output)

        api_out = pd.concat(outputs, ignore_index=True)
        api_out.set_index("id", inplace=True)
        merged_data = api_out.join(self.known_data, on="id", lsuffix="_api", rsuffix="_known")
        merged_data.replace({'female': 'F', 'male': 'M'}, inplace=True)
        merged_data.to_csv(self.output_file)
        return merged_data

    def _query_batch(self, data, lower, upper):
        subset = data[lower:upper]
        payload_data = [x for x in subset[["name", "id"]].T.to_dict().values()]
        payload = {"personalNames": payload_data}
        response = requests.request("POST", self._URL, json=payload, headers=self._HEADERS)
        return pd.DataFrame(response.json()['personalNames'])


if __name__ == "__main__":
    runner = NameSorRunner(NAMSOR_URL, OUTPUT_FILE, KNOWN_DATA_FILE)
    runner.main()
