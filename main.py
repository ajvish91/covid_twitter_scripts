import unicodedata
import string
import re
import warnings
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import gensim
import sys
import os
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
from pprint import pprint
np.random.seed(2018)
stemmer = SnowballStemmer("english")
# Enable logging for gensim - optional
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


new_stopwords = []

if os.path.isfile("stopwords.txt"):
    with open("stopwords.txt") as f:
        new_stopwords = f.readlines()
    new_stopwords = [i.rstrip("\n") for i in new_stopwords]
my_stop_words = STOPWORDS.union(set(new_stopwords))

tqdm.pandas()


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_non_ascii(words, encoding):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.replace('â', '')
        if encoding == "latin1":
            new_word = new_word.encode("iso-8859-1").decode("iso-8859-1")
        new_word = unicodedata.normalize('NFKD', new_word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def normalize(words, encoding):
    words = remove_non_ascii(words, encoding)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    return words


def preprocess(text ,encoding):
    text = remove_URL(text)
    text = " ".join(normalize(text.split(" "), encoding))
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in my_stop_words and len(token) > 1:
            #             lem = lemmatize_stemming(token)
            #             if lem not in my_stop_words:
            result.append(token)
    return result


def process_ngrams(texts, bigram_mod, trigram_mod, stop_words=STOPWORDS, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
#     texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in tqdm(texts)]
    texts = [trigram_mod[bigram_mod[doc]] for doc in tqdm(texts)]
    return texts


def main():
    pprint(sys.argv)
    if len(sys.argv) != 6:
        print("Error: Please run the script in the following format:\n")
        print("\tpython main.py <data filename> <data column name> <sample> <encoding> <num topics>\n")
        print("Example: \"python main.py tweets.xlsx Tweet 0.01 utf8 10\"")
        return

    path = sys.argv[1]
    column = sys.argv[2]
    sample = sys.argv[3]
    encoding = sys.argv[4]
    topics = sys.argv[5]

    if not os.path.isfile(path):
        pprint("Error: data file not found in " + path)
        return

    if not path.endswith(".xlsx") and not path.endswith(".csv"):
        pprint("Please select an excel (.xlsx) or csv (.csv) data file")
        return

    if encoding != "utf8" and encoding != "latin1":
        pprint("Please choose either 'utf8' or 'latin1' as encoding")
        return

    try: 
        float(sample)
    except ValueError:
        print("please enter a fraction less than or equal to 1")
        return
    
    if float(sample) > 1:
        print("please enter a fraction less than or equal to 1")
        return

    try: 
        float(topics)
    except ValueError:
        print("please enter an integer greater than 2 or less than 101")
        return
    
    if float(topics) < 3 or float(topics) > 101:
        print("please enter an integer greater than 2 or less than 101")
        return

    pprint("Reading data...")
    if path.endswith(".xlsx"):
        data = pd.read_excel(path, encoding=encoding)
    else:
        data = pd.read_csv(path, encoding=encoding)
    pprint(data)

    if column not in data.columns:
        pprint("Error: please enter valid column name. '" +
               column + "' was entered.")
        return
    pprint(column)

    data.dropna(subset=[column], inplace=True)
    data.reset_index(inplace=True)
    pprint("Preprocessing data...")
    documents = data.loc[:, [column]]
    documents['index'] = data.index

    processed_docs = documents[column].progress_map(lambda x: preprocess(x, encoding))

    pprint("generating bigrams")
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(processed_docs, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    pprint('generating trigrams')
    trigram = gensim.models.Phrases(bigram[processed_docs], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    pprint("adding bigrams and trigrams to data")
    processed_docs = process_ngrams(
        processed_docs, bigram_mod=bigram_mod, trigram_mod=trigram_mod)

    pprint("createing a dictionary in corpus")
    dictionary = gensim.corpora.Dictionary(processed_docs[:1000000])
    if len(processed_docs) > 1000000:
        for i in tqdm(range(1000000, len(processed_docs), 1000000)):
            dictionary.add_documents(processed_docs[i:i+100000])

    pprint("filtering extremes")
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    pprint("saving dictionary")
    dictionary.save(path + "_dict.gensim")

    pprint("creating bag of words corpus")
    bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(processed_docs)]

    pprint("saving")
    pickle.dump(bow_corpus, open(path + '_bow_corpus.pkl', 'wb'))

    if float(sample) < 1:
        sampled = data.sample(frac=float(sample)).reset_index()

        sample_documents = sampled.loc[:, [column]]
        sample_documents['index'] = sampled.index
        print(sample_documents.head())
        sample_processed_docs = sample_documents[column].progress_map(lambda x: preprocess(x, encoding))
        print(sample_processed_docs[:10])
        # higher threshold fewer phrases.
        bigram = gensim.models.Phrases(
            sample_processed_docs, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(
            bigram[sample_processed_docs], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        sample_processed_docs = process_ngrams(sample_processed_docs, bigram_mod, trigram_mod)
        print("bigrams done")
        sampled_bow_corpus = [dictionary.doc2bow(
            doc) for doc in tqdm(sample_processed_docs)]
        sample_dictionary = gensim.corpora.Dictionary(sample_processed_docs)
    else:
        sampled_bow_corpus = bow_corpus
        sample_documents = documents
        sample_processed_docs = processed_docs
        sample_dictionary = dictionary

    print("Training...")
    lda_model = gensim.models.LdaMulticore(sampled_bow_corpus,
                                           num_topics=topics,
                                           id2word=dictionary,
                                           passes=10,
                                           alpha="asymmetric",
                                           workers=29,
                                           eta=0.9099999999999999,
                                           random_state=293)
    lda_model.save("model_sample_1006_1600.gensim")

    for idx, topic in lda_model.print_topics(-1, num_words=15):
        print('Topic: {} Words: {}'.format(idx, topic))

    # Compute Perplexity
    perlexity = lda_model.log_perplexity(sampled_bow_corpus)
    # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', perlexity)
    print("\nEvaluating model...")

    # Compute Coherence Score
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
                                                       texts=sample_processed_docs,
                                                       dictionary=dictionary,
                                                       coherence='c_v',
                                                       processes=16)
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_topics = coherence_model_lda.get_coherence_per_topic()
    print('\nAverage Coherence Score: ', coherence_lda)
    print('\nTopic-wise Coherence Score: ', coherence_topics)

    print("\nMapping topics with each row...")

    d = [{"document_id": doc_id,
      "document": documents[column][doc_id],
      "corpus_bow": doc,
      "topic_distribution": lda_model[doc]} for doc_id, doc in tqdm(enumerate(bow_corpus)) if doc is not None]
    for i, kv in enumerate(d):
        topics_dist = kv["topic_distribution"]
        del kv["topic_distribution"]
        for i in range(int(topics)):
            kv["topic " + str(i)] = False
        for top in topics_dist:
            key = "topic " + str(top[0])
            if len(kv["corpus_bow"]) == 0:
                kv[key] = False
            elif top[1] < 1.0/int(topics):
                kv[key] = False
            else:
                kv[key] = True
        del kv["corpus_bow"]

    map_df = pd.DataFrame(d)
    to_save = pd.concat([data, map_df], axis=1).drop(columns=["document", "document_id"])
    print("Saving...")
    topic_list = []
    for idx, topic in lda_model.print_topics(-1, num_words=15):        
        word_pairs =  [pair.rstrip(" ").split("*") for pair in topic.split("+")]
        words = []
        probs = []
        for w in word_pairs:
            words.append(w[1].strip("\""))
            probs.append(w[0])
        topic_dict = {}
        topic_dict["Topic"] = idx        
        topic_dict["Coherence Score"] = coherence_topics[idx]
        for i in range(len(words)):
            topic_dict["Word " + str(i)] = words[i]
        for i in range(len(probs)):
            topic_dict["Relevance " + str(i)] = probs[i]
        topic_list.append(topic_dict)
    topic_df = pd.DataFrame(topic_list)
    writer = pd.ExcelWriter(path[:-5] + '_topics.xlsx', engine='xlsxwriter', options={'strings_to_urls': False})
    to_save.to_excel(writer, sheet_name='Data')
    topic_df.to_excel(writer, sheet_name='Topics')
    writer.save()


if __name__ == "__main__":
    main()