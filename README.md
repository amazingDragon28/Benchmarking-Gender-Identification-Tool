# Benchmarking Gender Identification Tool

This project benchmarks different gender identification tools on the PubMed and arXiv data set, and generates a Silver Standard Corpora based on inferred gender on arXiv. 

The description of each file and folder is as following:

- data: The directory contains data needed to be load and when running program. 
  - author_10000.csv: The most frequent 10000 first names in the arXiv data.
  - author_unique.csv: All unique first names in the arXiv data.
  - author_without_initial.7z: All names in the arXiv data after dropping initials and non-human names.
  - authors.7z: All author names in arXiv data.
  - pubmed.csv: author names from PubMed, contained manaully labeled gender.

- envs: All environment requirements used in this project, including conda environment and docker environment.
  
- output: All output files saved in this directory. Some special data files are explained below.
  - gender_corpora_10000.csv: This csv file contains the assigned gender and confidence score based on voting mechansim, gender results from four tools and their confidence score for the most frequent 10000 first names in the arXiv data. 
  - no_author_paper_id.txt: The txt file containing paper ids of all papers that have no author with an identified gender. 
  - silver_corpora.7z: The final Silver Standard Corpora (papers that have no author with an identified gender are removed).

- scripts: This directory contains all scripts.
  - nam_sor_analysis.py: The code to query NamSor to guess gender.
  - run_benchmark_arxiv.py: The code to benchmark four tools on arXiv data set and generate the silver standard corpora.
  - run_benchmark_pubmed.py: The code to benchmark six tools on PubMed data set.
  - run_genderapi_example.py: The code to query Gender Api to guess gender (need API key).
  - run_genderdetector_example.py: The code to query gender detector to guess a gender.
  - run_genderguesser_example.py: The code to query gender-guesser.
  - run_genderize_example: The code to query genderize.io to guess a gender.
  - run_get_name.py: Obtain author parsed names and paper id from arXiv data set stored in MongoDB.
  - run_nameapi_example: The code to request Name API to predict a gender (need API key).

NOTE: 
- Due to the limit of size file on GitHub, all large data files are compressed as 7z files. All csv files with 7z suffix should be unpacked before running scripts. 
- You can simply run following code to observe the benchmark results on PubMed and arXiv data set.

    ```python
    python run_benchmark_pubmed.py
    python run_benchmark_arXiv.py
    ```
- My supervisor sets Docker environment and stores the arXiv data set in the Docker container. The code for runnning NamSor is a reuse of Stefan's code in the gender-disparity project. Therefore, the code of these parts should not be marked.  
- I provide my API key for Gender API, Name API and NamSor, so you can use it to test how these tools are used to guess a gender. You can also apply your own API keys and replace them on corresponding files. These three tools are not unlimited, so please pay attention to the number of credits you have used.

