from io import open

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import bigrams

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
def generate_bigrams_with_start_and_end(train_sentences):
    tags = []
    for sent in train_sentences:
        tags.append('<s>')
        for token in sent:
            tags.append(token['upos'])
        tags.append('</s>')
    tags_bigrams = list(bigrams(tags))
    return tags_bigrams


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
def tansition_using_witten_bell_smoothing(tags_bigram):
    smoothed = {}
    distinct_tags = set([t for (t, _) in tags_bigram])
    for tag1 in distinct_tags:
        tag2 = [t2 for (t1, t2) in tags_bigram if t1 == tag1]
        smoothed[tag1] = WittenBellProbDist(FreqDist(tag2), bins=1e5)
    return smoothed


# calculates the transition probability for tags
def calculate_transition_prob(witten_trans_smooth, tag1, tag2):
    return witten_trans_smooth[tag1].prob(tag2)


def greedy_viterbi(test_sentences, train_word_tags, train_tags_bigram):
    smoothed_trans = tansition_using_witten_bell_smoothing(train_tags_bigram)
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

                # << TRANSITION PROB CALCULATION >>
                if key == 0:  # condition for starting tag <s>
                    alpha = calculate_transition_prob(smoothed_trans, '<s>', tag)
                else:
                    # take the last element, which is always the previous added tag with max V
                    alpha = calculate_transition_prob(smoothed_trans, predicted_tags[len(predicted_tags) - 1], tag)

                # <<EMISSION PROB CALCULATION
                beta = calculate_emission_probability(smoothed_emission, word, tag)
                tag_prob = alpha * beta
                tag_prob_list_per_word.append(tag_prob)

            max_tag_prob = max(tag_prob_list_per_word)
            max_tag = tags[tag_prob_list_per_word.index(max_tag_prob)]
            predicted_tags.append(max_tag)

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

    print("\nGreedy Algorithm:")
    predicted_train_tags = greedy_viterbi(train_sents, get_words_tags(train_sents),
                                          generate_bigrams_with_start_and_end(train_sents))
    actual_train_tags = get_words_tags(train_sents)
    print("Accuracy on train: {}%".format(calculate_accuracy(predicted_train_tags, actual_train_tags) * 100))

    predicted_test_tags = greedy_viterbi(test_sents, get_words_tags(test_sents),
                                         generate_bigrams_with_start_and_end(test_sents))
    actual_test_tags = get_words_tags(test_sents)
    print("Accuracy on test: {}%".format(calculate_accuracy(predicted_test_tags, actual_test_tags) * 100))
