"""
Script to generate small test data set to be used in the absence of a decision over the test corpus/corpora. Uses Faker
to generate a specified number of male and female names. Furthermore requires an output file path to store results.

Usage from root directory of the project:

    python scripts/generate_faker_data.py --output data/test_names.csv --number 1000

"""
import argparse
import sys

from equigen.utils.generate_test_data import check_validity_input
from equigen.utils.generate_test_data import generate_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender disparity test data")
    parser.add_argument('--output', dest='output', help="Path to where output file is to be stored")
    parser.add_argument('--number', dest='number', help="Number of test objects (both female and male)")
    args = parser.parse_args()

    msg, validity = check_validity_input(args)

    if validity:
        test_subjects = generate_data(int(args.number))
        with open(args.output, "w") as f:
            for i in test_subjects:
                f.write(f"{i[0]}\t{i[1]}\n")
    else:
        print(msg)
        sys.exit(1)
