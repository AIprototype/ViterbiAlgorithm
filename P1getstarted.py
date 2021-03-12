from io import open

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import ngrams

from conllu import parse_incr, TokenList
import pandas as pd
from collections import Counter

corpora = {}
corpora['en'] = 'UD_English-EWT/en_ewt'
corpora['es'] = 'UD_Spanish-GSD/es_gsd'
corpora['nl'] = 'UD_Dutch-Alpino/nl_alpino'


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


def get_wordTagList_total_tags_total_tokens(data_set):
    words_and_tags = []
    tokens_list = []
    tags_list = []
    for sent in train_sents:
        for token in sent:
            word_and_tag = [token['form'], token['upos']]
            words_and_tags.append(word_and_tag)
            tokens_list.append(token['form'])
            tags_list.append(token['upos'])
    return words_and_tags, tokens_list, tags_list


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

    # Illustration how to access the word and the part-of-speech of tokens.
    # for sent in train_sents:
    #     for token in sent:
    #         print(token['form'], '->', token['upos'], sep='', end=' ')
    #     print()

    # first sentence in the train
    pos_tags, tokens, tags = get_wordTagList_total_tags_total_tokens(train_sents)
    unique_tags = list(set(tags))
    unique_words = list(set(tokens))
