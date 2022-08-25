import pandas as pd
import gender_guesser.detector as gender

from pathlib import Path

TEST_FILE = Path(__file__).parent.joinpath("../data/test_names_1000.csv")

if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)
    dec = gender.Detector()

    for name in test_data['Name']:
        print(dec.get_gender(name.split(" ")[0]))
