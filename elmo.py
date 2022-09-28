import string

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from reader import DataReader


# Calculates cosine similarity
def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


def sent_tokenize(text):
    sents = nlkt_sent_tokenize(text)
    sents_filtered = []
    for s in sents:
        sents_filtered.append(s)
    return sents_filtered


def cleanup_sentences(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    sentences_cleaned = []
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if not w.lower() in stop_words]
        words = [w.lower() for w in words]
        sentences_cleaned.append(" ".join(words))
    return sentences_cleaned


def get_tf_idf(sentences):
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()

    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names_out()

    relevant_vector_indices = np.where(centroid_vector > 0.3)[0]

    word_list = list(np.array(feature_names)[relevant_vector_indices])
    return word_list


def word_vectors_cache(sentences, embedding_model):
    # Create an instance of the Elmo model
    embeddings = embedding_model(sentences, signature="default", as_dict=True)["elmo"]
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    word_vectors = dict()
    for sent_index, sent in enumerate(sentences):
        words = nlkt_word_tokenize(sent)
        for index, w in enumerate(words):
            word_embedding = sess.run(embeddings[sent_index][index])
            # print(word_embedding)
            word_vectors.update({w: word_embedding})

    return word_vectors


def word_vectors_cache_opt(sentences, embedding_model):
    # Create an instance of the Elmo model
    embeddings = embedding_model(sentences, signature="default", as_dict=True)["elmo"]
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    word_vectors = dict()
    for sent_index, sent in enumerate(sentences):
        words = nlkt_word_tokenize(sent)
        inputs = []
        for index, w in enumerate(words):
            inputs.append(embeddings[sent_index][index])
        word_embeddings = sess.run(inputs)
        for index, w in enumerate(words):
            word_embedding = word_embeddings[index]
            word_vectors.update({w: word_embedding})

    return word_vectors


# Sentence embedding representation with sum of word vectors
def build_embedding_representation(words, word_vectors, embedding_model):
    embedding_representation = np.zeros(1024, dtype="float32")
    word_vectors_keys = set(word_vectors.keys())
    count = 0
    for w in words:
        if w in word_vectors_keys:
            embedding_representation = embedding_representation + word_vectors[w]
            count += 1
    if count != 0:
        embedding_representation = np.divide(embedding_representation, count)
    return embedding_representation


def summarize(text, emdedding_model):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    centroid_words = get_tf_idf(clean_sentences)
    word_vectors = word_vectors_cache_opt(clean_sentences, emdedding_model)
    # print(word_vectors)
    # Centroid embedding representation
    centroid_vector = build_embedding_representation(centroid_words, word_vectors, emdedding_model)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        scores = []
        words = clean_sentences[i].split()

        # Sentence embedding representation
        sentence_vector = build_embedding_representation(words, word_vectors, emdedding_model)

        # Cosine similarity between sentence embedding and centroid embedding
        score = similarity(sentence_vector, centroid_vector)
        sentences_scores.append((i, raw_sentences[i], score, sentence_vector))
    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    count = 0
    sentences_summary = []
    # Handle redundancy
    for s in sentence_scores_sort:
        if count > 100:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = similarity(s[3], ps[3])
            if sim > 0.95:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += len(s[1].split())

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

    summary = "\n".join([s[1] for s in sentences_summary])
    return summary


if __name__ == '__main__':

    reader = DataReader('/content/drive/MyDrive/NLP_Project/cnn_data.csv')
    documents, summaries = reader.load_dataset_from_csv()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_recalls = np.zeros(shape=(len(documents)))
    rouge_2_recalls = np.zeros(shape=(len(documents)))
    rouge_L_recalls = np.zeros(shape=(len(documents)))
    model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    for doc_num in range(2, len(documents)):
        try:
            # print('\n Document Number:',doc_num)
            document = documents[doc_num]
            summary = summaries[doc_num]
            clean_sentences = cleanup_sentences(document)
            words = []
            for sent in clean_sentences:
                words.append(nlkt_word_tokenize(sent))
            if len(words) > 0:
                generated_summary = summarize(document, model)
                scores = scorer.score(summary, generated_summary)
                rouge_1 = scores['rouge1'].recall
                rouge_2 = scores['rouge2'].recall
                rouge_L = scores['rougeL'].recall
                rouge_1_recalls[int(doc_num)] = rouge_1
                rouge_2_recalls[int(doc_num)] = rouge_2
                rouge_L_recalls[int(doc_num)] = rouge_L
                print(rouge_1, rouge_2, rouge_L)
        except Exception as e:
            pass

    print('Avg rouge 1 score {}'.format(np.average(rouge_1_recalls)))
    print('Avg rouge 2 score {}'.format(np.average(rouge_2_recalls)))
    print('Avg rouge L score {}'.format(np.average(rouge_L_recalls)))
