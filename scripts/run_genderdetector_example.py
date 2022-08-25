from gender_detector import gender_detector as gd
import pandas as pd


from pathlib import Path

TEST_FILE = Path(__file__).parent.joinpath("../data/test_names_1000.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output/gender_detector_output.csv")

if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)
    detector = gd.GenderDetector()
    
    output = pd.DataFrame()
    
    for name in test_data['Name']:
        try:
                output = pd.concat([output, pd.DataFrame.from_records([{"name": name, "gender": detector.guess(name.split(" ")[0])}])], ignore_index=True)
        except:
                output = pd.concat([output, pd.DataFrame.from_records([{"name": name, "gender": 'unknown'}])], ignore_index=True)

    output['gender'] = output['gender'].map({'female':'F','male':'M', 'unknown':'N'})

    if Path(OUTPUT_FILE).exists():
            output.to_csv(OUTPUT_FILE, mode='a', index = False, header=False)
    else:
            output.to_csv(OUTPUT_FILE, index = False)

    # # Calculate accurate
    diff = test_data['Gender'].compare(output['gender'])
    accuracy = (output.shape[0] - diff.shape[0]) / output.shape[0]
    print("The accuracy is {:.2%}".format(accuracy))

