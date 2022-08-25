import pandas as pd
from genderize import Genderize

from pathlib import Path

TEST_FILE = Path(__file__).parent.joinpath("../data/test_names_1000.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output/genderize_output.csv")

if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)
    # print(Genderize().get(test_data['Name'].str.split(r"\s|\.\s", expand=True)[0]))

    output = pd.DataFrame(Genderize().get(test_data['Name'].str.split(r"\s|\.\s", expand=True)[0]))
    output = output[['name', 'gender']]
    output['gender'] = output['gender'].map({'female':'F','male':'M', None:'N'})

    if Path(OUTPUT_FILE).exists():
            output.to_csv(OUTPUT_FILE, mode='a', index = False, header=False)
    else:
            output.to_csv(OUTPUT_FILE, index = False)

    # Calculate accurate
    diff = test_data['Gender'].compare(output['gender'])
    accuracy = (output.shape[0] - diff.shape[0]) / output.shape[0]
    print("The accuracy is {:.2%}".format(accuracy))