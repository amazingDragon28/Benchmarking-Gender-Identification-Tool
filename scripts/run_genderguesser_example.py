import pandas as pd
import gender_guesser.detector as gender
import re

from pathlib import Path

def run_gender_guesser(test_data, output_file):
    """
    Use the python package gender-guesser to guess gender from a person's 
    first name. You can create another case sensitive detector using `dec = gender.Detector()`
    """

    dec = gender.Detector(case_sensitive=False)
    output = pd.DataFrame()

    for name in test_data['first_name']:
        name = re.split("\s|\.\s", name)[0]
        output = pd.concat([output, pd.DataFrame({"api_gender": dec.get_gender(name), "api_gender_final": dec.get_gender(name)}, index=[0])], ignore_index=True)
    
    output['api_gender'] = output['api_gender_final'].map({'mostly_female':'female', 'mostly_male': 'male', 'andy': 'unknown'}).fillna(output['api_gender_final'])
    output = pd.concat([test_data, output], axis=1)

    output.to_csv(output_file, index = False)
