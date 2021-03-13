from io import open

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import bigrams
import numpy as np
from conllu import parse_incr
from sklearn.metrics import accuracy_score

corpora = {'en': 'UD_English-EWT/en_ewt', 'es': 'UD_Spanish-GSD/es_gsd', 'nl': 'UD_Dutch-Alpino/nl_alpino'}


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


# obtain the word and its tag from provided sentences
def get_words_tags(train_sentences):
    words, tags = [], []
    for sent in train_sentences:
        for token in sent:
            words.append(token['form'])
            tags.append(token['upos'])
    return list(zip(words, tags))  # returns a list of word tag tuple


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


def greedy_best_path(test_sentences, train_word_tags, train_tags_bigram):
    smoothed_trans = transition_using_witten_bell_smoothing(train_tags_bigram)
    smoothed_emission = emission_using_witten_bell_smoothing(train_word_tags)
    predicted_tags = []
    words_to_tag = []
    tags = list(set([t for w, t in train_word_tags]))
    for sent in test_sentences:
        words = []
        for token in sent:
            words.append(token['form'])
            words_to_tag.append(token['form'])

        # for each word in the sentence
        for key, word in enumerate(words):  # V(key)
            tag_prob_list_per_word = []
            for tag in tags:  # V(key)(tag)

                # << TRANSITION PROB CALCULATION (α) >>
                if key == 0:  # condition for starting tag <s>
                    alpha = calculate_transition_prob(smoothed_trans, '<s>', tag)
                else:
                    # take the last element, which is always the previous added tag with max V
                    alpha = calculate_transition_prob(smoothed_trans, predicted_tags[len(predicted_tags) - 1], tag)

                # <<EMISSION PROB CALCULATION (β) >>
                beta = calculate_emission_probability(smoothed_emission, word, tag)
                tag_prob = alpha * beta
                tag_prob_list_per_word.append(tag_prob)

            max_tag_prob = max(tag_prob_list_per_word)
            max_tag = tags[tag_prob_list_per_word.index(max_tag_prob)]
            predicted_tags.append(max_tag)

    return list(zip(words_to_tag, predicted_tags))


def viterbi_v2(test_sentences, train_word_tags, train_tags_bigram):
    smoothed_trans = transition_using_witten_bell_smoothing(train_tags_bigram)
    smoothed_emission = emission_using_witten_bell_smoothing(train_word_tags)
    predicted_tags = []
    words_to_tag = []
    tags = list(set([t for w, t in train_word_tags]))

    # test corpus is parsed using sentences with start: <s> and end: </s>
    for sent in test_sentences:
        words = []
        for token in sent:
            words.append(token['form'])
            words_to_tag.append(token['form'])

        col_length = len(words)
        row_length = len(tags)
        # 0th pos -> probability of current Vit, 1st pos -> index of prev tag
        viterbi_matrix = np.zeros((row_length, col_length))
        viterbi_best_tag_matrix = np.zeros((row_length, col_length))
        # scaling factor, for preventing underflow
        scale_matrix = np.zeros(col_length)

        # Part: 1 -> Initialise the viterbi matrix
        # for t=0
        for tag_index, tag in enumerate(tags):
            # transition prob of <s> to other tags
            alpha = calculate_transition_prob(smoothed_trans, '<s>', tag)
            # emission of the first word under the tag
            beta = calculate_emission_probability(smoothed_emission, words[0], tag)
            viterbi_matrix[tag_index, 0] = alpha * beta
            # the tag stored here isn't considered, as it is transitioning from <s>
            viterbi_best_tag_matrix[tag_index, 0] = 0
        # scaling
        scale_matrix[0] = 1 / np.sum(viterbi_matrix[:, 0])
        viterbi_matrix[:, 0] = scale_matrix[0] * viterbi_matrix[:, 0]

        # Part: 2 -> Middle transitions
        # starting from t>0
        for word_pos in range(1, len(words)):
            for curr_tag_pos, curr_tag in enumerate(tags):
                cell_trans_prob = []
                for prev_tag_pos, prev_tag in enumerate(tags):
                    # calculating all the alphas
                    alpha = calculate_transition_prob(smoothed_trans, prev_tag, curr_tag)  # alpha(q', q)
                    prob = viterbi_matrix[prev_tag_pos, word_pos - 1] * alpha  # v[q', t-1] * alpha(q', q)
                    cell_trans_prob.append(prob)
                # max alpha
                max_cell_trans_prob = max(cell_trans_prob)  # max_q'V[q', t-1] * alpha(q', q)
                # max alpha index, the tag position where it has max prob of transition
                max_cell_trans_index = cell_trans_prob.index(max_cell_trans_prob)

                beta = calculate_emission_probability(smoothed_emission, words[word_pos], curr_tag)  # beta(q, W_t)
                # V[q, t] = max_q'V[q', t-1] * alpha(q', q) * beta(q, W_t)
                viterbi_matrix[curr_tag_pos, word_pos] = max_cell_trans_prob * beta
                viterbi_best_tag_matrix[curr_tag_pos, word_pos] = max_cell_trans_index  # storing the tag pos
            # scaling
            scale_matrix[word_pos] = 1.0 / np.sum(viterbi_matrix[:, word_pos])  # scaling factor
            viterbi_matrix[:, word_pos] = scale_matrix[word_pos] * viterbi_matrix[:, word_pos]

        # Part: 3 -> Ending probability calculation, tags -> </s>
        # t = n + 1
        # V[q_f, n+1] = max_q'V[q', n].alpha(q', q_f)
        # here im storing it in a separate array called 'final_prob_list'
        final_prob_list = []
        for prev_tag_pos, prev_tag in enumerate(tags):
            alpha = calculate_transition_prob(smoothed_trans, prev_tag, '</s>')
            final_prob_list.append(viterbi_matrix[prev_tag_pos, (len(words) - 1)] * alpha)
        # custom scaling
        custom_scale = 1 / np.sum(final_prob_list)
        final_prob_list = [final_prob * custom_scale for final_prob in final_prob_list]
        # getting max and its index
        max_final_prob = max(final_prob_list)
        max_final_prob_index = final_prob_list.index(max_final_prob)

        # Part: 4 -> Backtracking
        # from the last word in the sentence, up to the first
        # so the tags of the last word will be the first in the backtrack list
        back_track_list = []
        predicted_tag_index = max_final_prob_index
        back_track_list.append(tags[predicted_tag_index])
        for word_pos in range(len(words) - 1, 0, -1):
            predicted_tag_index = int(viterbi_best_tag_matrix[predicted_tag_index, word_pos])
            back_track_list.append(tags[predicted_tag_index])
        back_track_list.reverse()
        for tag in back_track_list:
            predicted_tags.append(tag)

    print("{} {}".format(len(predicted_tags), len(words_to_tag)))
    return list(zip(words_to_tag, predicted_tags))


def calculate_accuracy(predicted_tags, actual_tags):
    predicted_tags = [predicted_tag[1] for predicted_tag in predicted_tags]
    actual_tags = [actual_tag[1] for actual_tag in actual_tags]
    return accuracy_score(actual_tags, predicted_tags)


if __name__ == '__main__':
    # Choose language.
    lang = 'en'

    # Limit length of sentences to avoid underflow.
    max_len = 100

    train_sents = conllu_corpus(train_corpus(lang))
    test_sents = conllu_corpus(test_corpus(lang))
    test_sents = [sent for sent in test_sents if len(sent) <= max_len]
    print(len(train_sents), 'training sentences')
    print(len(test_sents), 'test sentences')

    print("\nViterbi Algorithm:")
    train_word_tag = get_words_tags(train_sents)
    train_tag_bigrams = generate_bigram_list(train_sents)
    predicted_test_tags = viterbi_v2(test_sents,
                                     train_word_tag,
                                     train_tag_bigrams)
    actual_test_tags = get_words_tags(test_sents)
    print("Accuracy on test: {}%".format(calculate_accuracy(predicted_test_tags, actual_test_tags) * 100))

    print("\nGreedy Algorithm:")
    predicted_test_tags = greedy_best_path(test_sents, get_words_tags(train_sents),
                                           generate_bigram_list(train_sents))
    actual_test_tags = get_words_tags(test_sents)
    print("Accuracy on test: {}%".format(calculate_accuracy(predicted_test_tags, actual_test_tags) * 100))
