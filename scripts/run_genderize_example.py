import pandas as pd
from genderize import Genderize

from pathlib import Path

def run_genderize(test_data, output_file):
    """
    Query genderize without an API key. The package can only
    guess gender from first name.
    """

    output = pd.DataFrame(Genderize().get(test_data['first_name']))

    output = output[['gender', 'count', 'probability']]
    output['gender'] = output['gender'].fillna('unknown')
    output = output.rename(columns={'gender': 'api_gender', 'count': 'api_count', 'probability': 'api_probability'})

    output = pd.concat([test_data, output], axis=1)

    output.to_csv(output_file, index = False)
