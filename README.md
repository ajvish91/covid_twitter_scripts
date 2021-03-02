# Topic Modelling on COVID-19 based tweets
This is a script accompanying the research article: **COVID-19 Twitter dataset with latent topics, sentiments and emotions attributes** by Raj Kumar et al. It is used to extract latent topics from a Twitter dataset. 

The input can be an .xlsx or .csv file containing a column of tweets. Since the algorithm is based on Latent Dirichlet Allocation, the number of topics need to be mentioned while running the script. Words to ignore in the dataset can be specified in the stopwords.txt file. The output of the program is a data file (.xlsx) containing the dataset with each row tagged with a topic, and, a sheet containing the word distribution within a given topic. 

### Steps to install python packages and run script

1. Unzip the file topic_modelling_covid_twitter.zip
2. Install the latest version of **python** (>=3.6) or create a conda virtual environment.
3. Open Command Prompt or Terminal depending on operating system (Windows, Linux or Mac OS)
4. Navigate to **./topic_modelling_covid_twitter** where ever it unzipped using `cd`
5. `pip install -r requirements.txt`
6. `python main.py <datafile> <column_name> <sample> <encoding> <num topics>`
    `<datafile>` can be .xlsx or .csv
    `<column_name>` should be a valid column in the data file
    `<sample>` is between 0 and 1 including 1
    `<encoding>` could be either `utf8` or `latin1`
    `<num topics>` is between 3 and 100

    **Example**: `python main.py tweets.xlsx Tweet 0.01 utf8 10`   

### Visualization
Can be done using HighCharts. Example: https://jsfiddle.net/5xa4Ld2g/

### Citation:
If you use this script and find it useful for your research, please cite the source as:
Gupta, R., Vishwanath, A., and Yang, Y. (2020), COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes, Preprint at: https://arxiv.org/abs/2007.06954 

For correspondence, please contact yangyp@ihpc.a-star.edu.sg
