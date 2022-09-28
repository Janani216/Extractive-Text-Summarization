import string

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizer, AutoTokenizer, BertTokenizerFast
import concurrent.futures

from reader import DataReader


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")

    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)


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


# Populate word vector with all embeddings.
# This word vector is a look up table that is used
# for getting the centroid and sentences embedding representation.
def word_vectors_cache(sentences, embedding_model, tokenizer, layers):
    word_vectors = dict()
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        for idx in range(len(words)):
            w = words[idx]
            word_embedding = get_word_vector(sent, idx, tokenizer, embedding_model, layers)
            word_vectors.update({w: word_embedding.numpy()})
    return word_vectors


# Sentence embedding representation with sum of word vectors
def build_embedding_representation(words, word_vectors):
    embedding_representation = np.zeros(768, dtype="float32")
    word_vectors_keys = set(word_vectors.keys())
    count = 0
    for w in words:
        if w in word_vectors_keys:
            embedding_representation = embedding_representation + word_vectors[w]
            count += 1
    if count != 0:
        embedding_representation = np.divide(embedding_representation, count)
    return embedding_representation


def summarize(text, emdedding_model, tokenizer, layers):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    centroid_words = get_tf_idf(clean_sentences)
    word_vectors = word_vectors_cache(clean_sentences, emdedding_model, tokenizer, layers)
    # Centroid embedding representation
    centroid_vector = build_embedding_representation(centroid_words, word_vectors)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        words = clean_sentences[i].split()

        # Sentence embedding representation
        sentence_vector = build_embedding_representation(words, word_vectors)

        # Cosine similarity between sentence embedding and centroid embedding
        score = similarity(sentence_vector, centroid_vector)
        sentences_scores.append((i, raw_sentences[i], score, sentence_vector))
    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    # for s in sentence_scores_sort:
    #     print(s[0], s[1], s[2])
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


def generate_summary_for_document(inputs):
    document = inputs[0]
    summary = inputs[1]
    model = inputs[2]
    tokenizer = inputs[3]
    layers = inputs[4]

    try:
        clean_sentences = cleanup_sentences(document)
        words = []
        for sent in clean_sentences:
            words.append(nlkt_word_tokenize(sent))
        if len(words) > 0:
            generated_summary = summarize(document, model, tokenizer, layers)
            scores = scorer.score(summary, generated_summary)
            rouge_1 = scores['rouge1'].recall
            rouge_2 = scores['rouge2'].recall
            rouge_L = scores['rougeL'].recall
            print(rouge_1,rouge_2,rouge_L)
            return [rouge_1, rouge_2,rouge_L]
    except Exception as e:
      print(str(e))
      pass


if __name__ == '__main__':
    reader = DataReader('cnn_data.csv')
    documents, summaries = reader.load_dataset_from_csv()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
    documents = documents[468:]
    summaries = summaries[468:]
    rouge_1_recalls = []
    rouge_2_recalls = []
    rouge_L_recalls = []
    results = []
    layers = [-4, -3, -2, -1]
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained(
        '/content/drive/MyDrive/NLP_Project/bert_model/Fine_Tuned_Bert_5Epochs', local_files_only=True)
    all_inputs = []
    for doc_num in range(len(documents)):
      all_inputs.append((documents[doc_num], summaries[doc_num], model, tokenizer,
                                layers));

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   results = executor.map(generate_summary_for_document,all_inputs);

    count = 468
    for all_input in all_inputs:
      print("Executing for count ",count)
      count = count +1
      results.append(generate_summary_for_document(all_input))
    
    for result in results:
        rouge_1_recalls.append(results[0])
        rouge_2_recalls.append(results[1])
        rouge_L_recalls.append(results[2])
    print('Avg rouge 1 score {}'.format(np.average(np.array(rouge_1_recalls))))
    print('Avg rouge 2 score {}'.format(np.average(np.array(rouge_2_recalls))))
    print('Avg rouge L score {}'.format(np.average(np.array(rouge_L_recalls))))

