import os
import pandas as pd
import requests
from gender_detector import gender_detector as gd
import gender_guesser.detector as gender
from genderize import Genderize
import numpy as np

from pathlib import Path
from run_genderapi_example import run_gender_api
from run_genderdetector_example import run_gender_detector
from run_genderguesser_example import run_gender_guesser
from run_genderize_example import run_genderize
from run_nameapi_example import run_name_api
from name_sor_analysis import run_name_sor

GENDER_API_KEY = 'NNnBHUjAS2g3bEaQj5TLR3NDqcScm8hfAVXc'
GENDER_API_URL = "https://gender-api.com/v2/gender"

NAME_API_KEY = '843bd95c449fa0a67b2a9873bfa476b3-user1'
NAME_API_URL = "https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey=" + NAME_API_KEY

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
NAMSOR_KEY = '3443475a9c6cdc01197493f47d70b21c'

TEST_FILE = Path(__file__).parent.joinpath("../data/test.csv")
OUTPUT_FILE = Path(__file__).parent.joinpath("../output")

def cal_metrics(data_file):
    api_data = pd.read_csv(data_file)
    mm_df = np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'male'))
    mm = mm_df[0].size

    mf_df = np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'female'))
    mf = mf_df[0].size

    mu_df= np.where((api_data['gender'] == 'male') & (api_data['api_gender'] == 'unknown'))
    mu = mu_df[0].size

    fm_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'male'))
    fm = fm_df[0].size

    ff_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'female'))
    ff = ff_df[0].size

    fu_df = np.where((api_data['gender'] == 'female') & (api_data['api_gender'] == 'unknown'))
    fu = fu_df[0].size

    errorCoded = (fm + mf + mu + fu) / (mm + fm + mf + ff + mu + fu)
    errorCodedWithoutNA = (fm + mf) / (mm + fm + mf + ff)
    naCoded = (mu + fu) / (mm + fm + mf + ff + mu + fu)
    errorGenderBias = (mf - fm) / (mm + fm + mf + ff)

    # print("errorCoded: %f" % errorCoded)
    # print("errorCodedWithoutNA: %f" % errorCodedWithoutNA)
    # print("naCoded: %f" % naCoded)
    # print("errorGenderBias: %f" % errorGenderBias)

    return [errorCoded, errorCodedWithoutNA, naCoded, errorGenderBias], [mm, mf, mu, fm, ff, fu]

    

if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)
    test_data = test_data.iloc[:, 0:5]
    test_data['gender'] = test_data['gender'].map({'f': 'female', 'm': 'male', 'u': 'unknown'})
    
    run_gender_api(test_data, str(OUTPUT_FILE) + '/gender_api_pubmed.csv')
    run_gender_detector(test_data, str(OUTPUT_FILE) + '/gender_detector_pubmed.csv')
    run_gender_guesser(test_data, str(OUTPUT_FILE) + '/gender_guesser_pubmed.csv')
    run_genderize(test_data, str(OUTPUT_FILE) + '/genderize_pubmed.csv')
    run_name_api(test_data, str(OUTPUT_FILE) + '/name_api_test.csv')
    run_name_sor(test_data, str(OUTPUT_FILE) + '/namsor_pubmed.csv')

    # print("gender api")
    gender_api_metrics, gender_api_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_api_pubmed.csv")

    # print("gender detector")
    gender_detector_metrics, gender_detector_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_detector_pubmed.csv")

    # print("gender guesser")
    gender_guesser_metrics, gender_guesser_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/gender_guesser_pubmed.csv")

    # print("genderize")
    genderize_metrics, genderize_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/genderize_pubmed.csv")

    # print("name api")
    name_api_metrics, name_api_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/name_api_pubmed.csv")

    # print("NamSor")
    namsor_metrics, namsor_confusion_matrix = cal_metrics(str(OUTPUT_FILE)+"/namsor_pubmed.csv")

    list_of_metrics = [gender_api_metrics, gender_detector_metrics, gender_guesser_metrics, genderize_metrics, name_api_metrics, namsor_metrics]
    metrics = pd.DataFrame(list_of_metrics, index=['Gender API', 'Gender Detector', 'Gender Guesser', 'Genderize', 'Name API', 'NamSor'], 
    columns=['errorCoded', 'errorCodedWithoutNA', 'naCoded', 'errorGenderBias'])

    print(metrics)
    metrics.to_csv(str(OUTPUT_FILE) + "/tools_performance_pubmed.csv")

    iterables = [['Gender API', 'gender-detector', 'gender-guesser', 'genderize.io', 'Name API', 'NamSor'], ['male', 'female']]
    index = pd.MultiIndex.from_product(iterables) 
    list_of_confusion_matrix = [gender_api_confusion_matrix[:3], gender_api_confusion_matrix[3:], gender_detector_confusion_matrix[:3], gender_detector_confusion_matrix[3:],
                                gender_guesser_confusion_matrix[:3], gender_guesser_confusion_matrix[3:], genderize_confusion_matrix[:3], genderize_confusion_matrix[3:], 
                                name_api_confusion_matrix[:3], name_api_confusion_matrix[3:], namsor_confusion_matrix[:3], namsor_confusion_matrix[3:]]
    confusion_matrix = pd.DataFrame(list_of_confusion_matrix, index=index, 
    columns=['Predicted as male', 'Predicted as female', 'Predicted as unknown'])

    print(confusion_matrix)  
    confusion_matrix.to_csv(str(OUTPUT_FILE) + "/tools_confusion_matrix_pubmed.csv") 
