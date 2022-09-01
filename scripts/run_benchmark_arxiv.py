import pandas as pd
from pathlib import Path
from genderize import Genderize
import re
from collections import Counter
import numpy as np
import pickle

from run_get_name import get_authors_name
from run_genderdetector_example import run_gender_detector
from run_genderguesser_example import run_gender_guesser
from run_genderize_example import run_genderize
from name_sor_analysis import run_name_sor

GENDERIZE_API_KEY = "a697e3a5320785902757e99ca495a41b"

NAMSOR_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderFullBatch"
NAMSOR_KEY = 'd6604db5c5a3fe8760a9bfefd0038132'

OUTPUT_FILE = Path(__file__).parent.joinpath("../output")
DATA_FILE = Path(__file__).parent.joinpath("../data")

def run_genderize(test_data, output_file):
    """
    Query genderize with API key for large-scale data. The value of API key can be edited by updating the
    global variable `GENDERIZE_API_KEY`. Notice that genderize can only query with first name. 

    Parameters
    -----------
    test_data: pandas DataFrame
        A DataFrame contained author first name, last name and full names.

    output_file: Path
        The path where the output file should be saved. 
    """
    
    genderize = Genderize(user_agent='GenderizeDocs/0.0', api_key=GENDERIZE_API_KEY)
    # output = pd.DataFrame(genderize.get(test_data['first_name'].str.split(r"\s|\.\s", expand=True)[0]))
    output = pd.DataFrame()
    for name in test_data['first_name']:
        output = pd.concat([output, pd.DataFrame(genderize.get([re.split("\s|\.\s", name)[0]]))], ignore_index=True)

    output = output[['gender', 'count', 'probability']]
    output['gender'] = output['gender'].fillna('unknown')
    output = output.rename(columns={'gender': 'api_gender', 'count': 'api_count', 'probability': 'api_probability'})

    test_data = test_data.reset_index(drop=True)
    output = pd.concat([test_data, output], axis=1)

    if Path(output_file).exists():
        output.to_csv(output_file, mode='a', index = False, header=False)
    else:
        output.to_csv(output_file, index = False)
    return output


def determine_gender(row):
    """
    Determine the final gender in the silver standard corpora for a name based on 
    predicted genders of four tools. If only one tool assigns the gender, return this
    gender. If at least one tools predict gender, return the majority of gender.
    """
    count = Counter(list(row))
    if count.most_common()[0][1] == 4:
        return count.most_common()[0][0], 1, 'all'
    
    if (count.most_common()[0][0] == 'unknown') and (count['male'] != count['female']):
        return count.most_common()[1][0], 0.25, [row[row == count.most_common()[1][0]].index[0]]
    elif (count.most_common()[0][0] == 'unknown') and (count['male'] == count['female']):
        return 'neutral', 0.25, (row[row == 'male'].index[0], row[row == 'female'].index[0])
    elif (count['male'] == count['female']):
        return 'neutral', count['male'] / 4, (list(row[row == 'male'].index.values) + list(row[row == 'female'].index.values))
    else:
        # print("test")
        return count.most_common()[0][0], count.most_common()[0][1] / 4, list(row[row == count.most_common()[0][0]].index.values)

def cal_confidence(row):
    """
    Calculate the confidence score for the assigned gender in the corpus.
    """

    tools_map = {
        'genderize_gender': 'genderize_probability',
        'ns_gender': 'ns_probability',
        'gd_gender': 0.87, # highest threshold used: 0.99, lowest threshold: 0.75
        'gg_gender': 0.75  
    }
    if (row[1] == 1) and (row[0] != 'unknown'):
        confidence = 1
    elif (row[1] == 1) and (row[0] == 'unknown'):
        confidence = 0
    else:
        probability = [genders[tools_map[x]][row.name] if (x == 'ns_gender') or (x == 'genderize_gender') else tools_map[x] for x in row[2]]
        confidence = row[1] * (sum(probability) / len(probability)) 
    return confidence

def expand_corpora(row):
    """
    Expand the summarized gender result for the most frequent 10000 names to all names in the arXiv data set.
    """
    authors.loc[row[2] == authors['first_name'], 'gender'] = row[4]
    authors.loc[row[2] == authors['first_name'], 'confidence'] = row[5]

def cal_metrics(data, api_name):
    """
    Compare the results of the silver standard corpora and four tools. Calculate the performance
    metrics for these tools.

    Parameters
    ------------
    data: pandas DataFrame
        A DataFrame with gender in the standard corpora and predicted by four tools. 
    
    api_name: string
        The API used to calculate the metrics, including 'gd' for gender detector, 
        'gg' for gender-guesser, 'genderize' for genderize.io, and 'ns' for NamSor.
    """
    api_gender = api_name+'_gender'
    mm_df = np.where((data['gender'] == 'male') & (data[api_gender] == 'male'))
    mm = mm_df[0].size

    mf_df = np.where((data['gender'] == 'male') & (data[api_gender] == 'female'))
    mf = mf_df[0].size

    mu_df= np.where((data['gender'] == 'male') & (data[api_gender] == 'unknown'))
    mu = mu_df[0].size

    fm_df = np.where((data['gender'] == 'female') & (data[api_gender] == 'male'))
    fm = fm_df[0].size

    ff_df = np.where((data['gender'] == 'female') & (data[api_gender] == 'female'))
    ff = ff_df[0].size

    fu_df = np.where((data['gender'] == 'female') & (data[api_gender] == 'unknown'))
    fu = fu_df[0].size

    # unknown = np.where((api_data['gender'] == 'unknown'))
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
    
    if Path(str(DATA_FILE) + '/author.csv').exists():
        authors = pd.read_csv(str(DATA_FILE) + '/author.csv')
    else:
        authors = get_authors_name()

    # drop rows with NaN first name
    authors = authors.dropna(subset=['first_name'])

    # drop rows with initials first name
    ini_index = authors[authors['first_name'].str.match(r'[A-Za-z]\.')].index
    authors = authors.drop(ini_index)
    authors = authors.reset_index(drop=True)
    authors.to_csv(str(DATA_FILE) + "/author_without_initial.csv", index=False)

    # get the most frequent 10000 names (order: frequency high -> low)
    authors_10000 = authors.groupby(['first_name']).size().to_frame('count').reset_index() \
             .sort_values('count', ascending=False) \
             .drop_duplicates(subset='first_name')

    authors_10000 = authors_10000[:10000].apply(lambda x: authors[authors['first_name'] == x[0]].iloc[0], axis=1)


    # sort the most frequent 10000 names as the order of original order in arXive data base 
    key = lambda row: row['full_name']
    authors_10000 = authors[authors.apply(key, axis=1).isin(authors_10000.apply(key, axis=1))]
    authors_10000 = authors_10000.reset_index(drop=True)
    authors_10000.to_csv(str(DATA_FILE) + '/author_10000.csv', index=False)
    

    # drop duplicate first names
    authors = authors.drop_duplicates(subset='first_name', keep="first")
    authors['full_name'] = authors.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis = 1)
    authors.to_csv(str(DATA_FILE) + "author_unique.csv", index=False)


    # authors = pd.read_csv(str(DATA_FILE) + "author_unique.csv")
    # authors_10000 = pd.read_csv(str(DATA_FILE) + 'author_10000.csv')

    # if you need to test tools on arXiv data, please uncomment following lines
    # output_gd = run_gender_detector(authors, str(OUTPUT_FILE) + 'gender_detector_arxiv.csv')
    # output_gg = run_gender_guesser(authors, str(OUTPUT_FILE) + 'gender_guesser_arxiv.csv')
    # output_genderize = run_genderize(authors, str(OUTPUT_FILE) + 'genderize_arxiv.csv')
    # output_namsor = run_name_sor(authors_10000, str(OUTPUT_FILE) + 'namsor_arxiv.csv')

    ## merge output generated by tools together 
    output_gd = pd.read_csv(str(OUTPUT_FILE)+"/gender_detector_arxiv.csv")
    output_gg = pd.read_csv(str(OUTPUT_FILE)+"/gender_guesser_arxiv.csv")
    output_genderize = pd.read_csv(str(OUTPUT_FILE)+"/genderize_arxiv.csv")
    output_ns = pd.read_csv(str(OUTPUT_FILE)+"/namsor_arxiv.csv")

    ## select 10000 most frequent names results from gender-detector, gender-guesser, genderize results
    output_gd_10000 = output_gd[output_gd.apply(key, axis=1).isin(output_ns.apply(key, axis=1))]
    output_gd_10000 = output_gd_10000.reset_index(drop=True)
    output_gg_10000 = output_gg[output_gg.apply(key, axis=1).isin(output_ns.apply(key, axis=1))]
    output_gg_10000 = output_gg_10000.reset_index(drop=True)
    output_genderize_10000 = output_genderize[output_genderize.apply(key, axis=1).isin(output_ns.apply(key, axis=1))]
    output_genderize_10000 = output_genderize_10000.reset_index(drop=True)

    # the result for all tools on 10000 names
    genders = pd.concat([authors_10000, output_gd_10000['api_gender'], output_gg_10000['api_gender'], output_genderize_10000['api_gender'], output_ns['api_gender'], output_genderize_10000['api_probability'], output_ns['api_probabilityCalibrated']], axis='columns')
    genders.columns = ['paper id', 'last_name', 'first_name', 'full_name', 'gd_gender', 'gg_gender', 'genderize_gender', 'ns_gender', 'genderize_probability', 'ns_probability']
    genders.to_csv(str(OUTPUT_FILE) + '/genders_10000.csv', index=False)
    
    ## build the silver standard corpora on arXiv 10000 most frequent name
    gender1 = genders.loc[:,'gd_gender':'ns_gender'].apply(determine_gender, axis=1, result_type='expand')
    genders.insert(4, 'gender', gender1[0])
    genders.insert(5, 'confidence', gender1.apply(cal_confidence, axis=1))
    genders.to_csv(str(OUTPUT_FILE) + '/gender_corpora_10000.csv', index=False)

    # expand gender result to all authors
    # genders = pd.read_csv('output/gender_corpora_10000.csv')
    if Path(str(OUTPUT_FILE) + '/authors_gender_corpora_10000.csv').exists():
        authors = pd.read_csv(str(OUTPUT_FILE) + '/authors_gender_corpora_10000.csv')
    else:
        genders.apply(expand_corpora, axis=1)
        authors['gender'] = authors['gender'].fillna('unknown')
        genders.to_csv(str(OUTPUT_FILE) + '/authors_gender_corpora_10000.csv', index=False)

    # authors_gender = pd.read_csv('output/authors_gender_corpora_10000.csv')
    authors_gender1 = authors.groupby(['paper id'])['confidence'].transform('sum')
    grouped = authors.groupby(['paper id'])['confidence']
    author_gender_group = grouped.sum()
    # get all unknwon paper id
    index = author_gender_group[author_gender_group == 0].index.to_list()

    # drop papers with no authors assigned a gender  
    genders = authors[~authors['paper id'].isin(index)]
    genders.to_csv(str(OUTPUT_FILE) + '/silver_corpora.csv', index = False)

    # write index to file
    with open(str(OUTPUT_FILE) + '/no_author_paper_id.txt', 'wb') as fp:
        pickle.dump(index, fp)

    ## compare the final result with each tools
    # print("gender detector")
    genders = pd.read_csv(str(OUTPUT_FILE) + '/gender_corpora_10000.csv')
    gender_detector_metrics, gender_detector_confusion_matrix = cal_metrics(genders, 'gd')

    # print("gender guesser")
    gender_guesser_metrics, gender_guesser_confusion_matrix = cal_metrics(genders, 'gg')

    # print("genderize")
    genderize_metrics, genderize_confusion_matrix = cal_metrics(genders, 'genderize')

    # print("NamSor")
    namsor_metrics, namsor_confusion_matrix = cal_metrics(genders, 'ns')

    list_of_metrics = [gender_detector_metrics, gender_guesser_metrics, genderize_metrics, namsor_metrics]
    metrics = pd.DataFrame(list_of_metrics, index=['Gender Detector', 'Gender Guesser', 'Genderize', 'NamSor'], 
    columns=['errorCoded', 'errorCodedWithoutNA', 'naCoded', 'errorGenderBias'])

    print(metrics)
    metrics.to_csv(str(OUTPUT_FILE) + "/tools_performance_arxiv_10000.csv")