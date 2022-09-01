from gender_detector import gender_detector as gd
import pandas as pd
import re

from pathlib import Path

def run_gender_detector(test_data, output_file):
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

    output.to_csv(output_file, index = False)