import string

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle

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
    word_vectors = dict()
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        for w in words:
            word_vectors.update({w: embedding_model.wv[w]})
    return word_vectors


def build_embedding_representation(words, word_vectors, embedding_model):
    embedding_representation = np.zeros(embedding_model.vector_size, dtype="float32")
    word_vectors_keys = set(word_vectors.keys())
    count = 0
    for w in words:
        if w in word_vectors_keys:
            embedding_representation = embedding_representation + word_vectors[w]
            count += 1
    if count != 0:
        embedding_representation = np.divide(embedding_representation, count)
    return embedding_representation


def summarize(text, emdedding_model, threshold):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    centroid_words = get_tf_idf(clean_sentences)
    word_vectors = word_vectors_cache(clean_sentences, emdedding_model)
    # Centroid embedding representation
    centroid_vector = build_embedding_representation(centroid_words, word_vectors, emdedding_model)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        words = clean_sentences[i].split()
        sentence_vector = build_embedding_representation(words, word_vectors, emdedding_model)
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
            if sim > threshold:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += len(s[1].split())

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

    summary = "\n".join([s[1] for s in sentences_summary])
    return summary


def partition_data(documents, summaries):
    return shuffle(documents, summaries, random_state=0)[:100]


def perform_validation(documents, summaries):
    document_subsets, summary_subsets = partition_data(documents, summaries)
    rouge_1_scores = {}
    rouge_l_scores = {}
    thresholds = [0.75, 0.85, 0.95]
    for threshold in thresholds:
        rouge_1s = []
        rouge_Ls = []
        print('Performing validation for Word2Vec summarizer for threshold {}'.format(threshold))
        for doc_num in range(len(document_subsets)):
            try:
                document = document_subsets[doc_num]
                summary = summary_subsets[doc_num]
                clean_sentences = cleanup_sentences(document)
                words = []
                for sent in clean_sentences:
                    words.append(nlkt_word_tokenize(sent))
                if len(words) > 0:
                    model = Word2Vec(words, min_count=1, sg=1, window=5)
                    generated_summary = summarize(document, model, threshold)
                    scores = scorer.score(summary, generated_summary)
                    rouge_1 = scores['rouge1'].recall
                    rouge_L = scores['rougeL'].recall
                    rouge_1s.append(rouge_1)
                    rouge_Ls.append(rouge_L)
            except Exception as e:
                pass

        rouge_1_scores[threshold] = np.mean(np.array(rouge_1s))
        rouge_l_scores[threshold] = np.mean(np.array(rouge_Ls))
    return rouge_1_scores, rouge_l_scores


if __name__ == '__main__':
    reader = DataReader('~/Desktop/cnn_data.csv')
    documents, summaries = reader.load_dataset_from_csv()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # print(perform_validation(documents, summaries))
    rouge_1_recalls = np.zeros(shape=(len(documents)))
    rouge_2_recalls = np.zeros(shape=(len(documents)))
    rouge_L_recalls = np.zeros(shape=(len(documents)))
    for doc_num in range(len(documents)):
        try:
            document = documents[doc_num]
            summary = summaries[doc_num]
            clean_sentences = cleanup_sentences(document)
            words = []
            for sent in clean_sentences:
                words.append(nlkt_word_tokenize(sent))
            if len(words) > 0:
                model = Word2Vec(words, min_count=1, sg=1, window=5)
                generated_summary = summarize(document, model, threshold=0.95)
                scores = scorer.score(summary, generated_summary)
                print(scores)
                rouge_1 = scores['rouge1'].recall
                rouge_2 = scores['rouge2'].recall
                rouge_L = scores['rougeL'].recall
                rouge_1_recalls[int(doc_num)] = rouge_1
                rouge_2_recalls[int(doc_num)] = rouge_2
                rouge_L_recalls[int(doc_num)] = rouge_L
        except Exception:
            pass

    print('Avg rouge 1 score {}'.format(np.average(rouge_1_recalls)))
    print('Avg rouge 2 score {}'.format(np.average(rouge_2_recalls)))
    print('Avg rouge L score {}'.format(np.average(rouge_L_recalls)))
