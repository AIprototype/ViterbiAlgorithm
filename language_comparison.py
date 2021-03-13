from io import open

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import bigrams
import numpy as np
from conllu import parse_incr
from sklearn.metrics import accuracy_score
import pandas as pd

corpora = {'en': 'UD_English-EWT/en_ewt',
           'es': 'UD_Spanish-GSD/es_gsd',
           'nl': 'UD_Dutch-Alpino/nl_alpino',
           'ar': 'UD_Arabic-PADT/ar_padt',
           'fr': 'UD_French-Sequoia/fr_sequoia'}


def train_corpus(lang):
    return corpora[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return corpora[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


# generating the bigram tuple for transition
def generate_bigram_list(train_sentences):
    tags_outer = []
    for sent in train_sentences:
        tags = ['<s>']
        for token in sent:
            tags.append(token['upos'])
        tags.append('</s>')
        tags = list(bigrams(tags))
        tags_outer.extend(tags)
    return tags_outer


# to generate the smoothing dictionary
def emission_using_witten_bell_smoothing(word_tag_tuple):
    smoothed = {}
    tags = set([t for (_, t) in word_tag_tuple])
    for tag in tags:
        words = [w for (w, t) in word_tag_tuple if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    return smoothed


# calculates the emission probabilities for the training sentences
def calculate_emission_probability(witten_emission_smooth, word_to_check, tag_to_check):
    return witten_emission_smooth[tag_to_check].prob(word_to_check)


# calculates the smoothing for given tags in order of -> (tag1, tag2)
def transition_using_witten_bell_smoothing(tags_bigram):
    smoothed = {}
    distinct_tags = set([t for (t, _) in tags_bigram])
    for tag1 in distinct_tags:
        tag2 = [t2 for (t1, t2) in tags_bigram if t1 == tag1]
        smoothed[tag1] = WittenBellProbDist(FreqDist(tag2), bins=1e5)
    return smoothed


# calculates the transition probability for tags
def calculate_transition_prob(witten_trans_smooth, tag1, tag2):
    return witten_trans_smooth[tag1].prob(tag2)


def calculate_accuracy(predicted_tags, actual_tags):
    predicted_tags = [predicted_tag[1] for predicted_tag in predicted_tags]
    actual_tags = [actual_tag[1] for actual_tag in actual_tags]
    return accuracy_score(actual_tags, predicted_tags)


def generate_word_tag_lemma_data_frame(sentence_list):
    word_tag_lemma = []
    max_word_count = 18000
    current_count = 0
    for sent in sentence_list:
        for token in sent:
            if current_count < max_word_count:
                current_count += 1
                word_tag_lemma.append([token['form'], token['lemma'], token['upos']])
            else:
                break
    return pd.DataFrame(word_tag_lemma, columns=['word', 'lemma', 'pos_tag'])


def calculate_word_to_lemma_ratio(word_tag_lemma_df, lang):
    # lower lemma and higher word count means rich morphology
    print(lang)
    print("Total data frame size: {}".format(word_tag_lemma_df.shape[0]))
    print("Unique words to unique lemma ratio: {}".format(word_tag_lemma_df.loc[:, 'word'].nunique() /
                                                          word_tag_lemma_df.loc[:, 'lemma'].nunique()))


def generate_transition_matrix(lang_bigram, lang, pos_tags):
    smoothed = transition_using_witten_bell_smoothing(lang_bigram)
    trans_prob_matrix = np.zeros((len(pos_tags), len(pos_tags)))
    for curr_tag_pos, curr_tag in enumerate(pos_tags):
        for prev_tag_pos, prev_tag in enumerate(pos_tags):
            trans_prob_matrix[curr_tag_pos, prev_tag_pos] = calculate_transition_prob(smoothed,
                                                                                      prev_tag,
                                                                                      curr_tag)
    # for each row, check the max value, and the index of that max value is the tag that comes after the row tag
    predicted_bigram = get_tag_probability(trans_prob_matrix)
    return [lang, predicted_bigram]


def get_tag_probability(trans_prob_matrix):
    # for each row, check the max value, and the index of that max value is the tag that comes after the row tag
    tag_prediction = []
    for curr_tag_pos in range(len(tags)):
        row_list_tag_prob = trans_prob_matrix[curr_tag_pos, :]
        max_tag_prob = max(row_list_tag_prob)
        max_tag_index = np.where(row_list_tag_prob == max_tag_prob)
        tag_pred = [tags[curr_tag_pos], tags[max_tag_index[0][0]]]
        tag_prediction.append(tag_pred)
    return tag_prediction


if __name__ == '__main__':
    fr_df = generate_word_tag_lemma_data_frame(conllu_corpus(train_corpus('fr')))
    calculate_word_to_lemma_ratio(fr_df, 'French')
    print("\n")

    en_df = generate_word_tag_lemma_data_frame(conllu_corpus(train_corpus('en')))
    calculate_word_to_lemma_ratio(en_df, 'English')
    print("\n")

    ar_df = generate_word_tag_lemma_data_frame(conllu_corpus(train_corpus('ar')))
    calculate_word_to_lemma_ratio(ar_df, 'Arabic')
    print("\n")

    es_df = generate_word_tag_lemma_data_frame(conllu_corpus(train_corpus('es')))
    calculate_word_to_lemma_ratio(es_df, 'Spanish')
    print("\n")

    nl_df = generate_word_tag_lemma_data_frame(conllu_corpus(train_corpus('nl')))
    calculate_word_to_lemma_ratio(nl_df, 'Dutch')
    print("\n")

    fr_bigrams = list(bigrams(fr_df.loc[:, 'pos_tag']))
    en_bigrams = list(bigrams(en_df.loc[:, 'pos_tag']))
    ar_bigrams = list(bigrams(ar_df.loc[:, 'pos_tag']))
    es_bigrams = list(bigrams(es_df.loc[:, 'pos_tag']))
    nl_bigrams = list(bigrams(nl_df.loc[:, 'pos_tag']))

    # only storing tags that are common in all our languages
    tags = list(set.intersection(*map(set, [fr_df.loc[:, 'pos_tag'],
                                            en_df.loc[:, 'pos_tag'],
                                            ar_df.loc[:, 'pos_tag'],
                                            es_df.loc[:, 'pos_tag'],
                                            nl_df.loc[:, 'pos_tag']])))

    list_trans_matrix = [generate_transition_matrix(fr_bigrams, 'French', tags),
                         generate_transition_matrix(en_bigrams, 'English', tags),
                         generate_transition_matrix(ar_bigrams, 'Arabic', tags),
                         generate_transition_matrix(es_bigrams, 'Spanish', tags),
                         generate_transition_matrix(nl_bigrams, 'Dutch', tags)]

    # To check the similarity in the next possible post tag after a previous one
    for first_trans_matrix in list_trans_matrix:
        for second_trans_matrix in list_trans_matrix:
            if first_trans_matrix != second_trans_matrix:
                print("\n{} vs {}".format(first_trans_matrix[0], second_trans_matrix[0]))
                print(calculate_accuracy(first_trans_matrix[1], second_trans_matrix[1]))
