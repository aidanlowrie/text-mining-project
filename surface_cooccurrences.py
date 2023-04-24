import nltk
from nltk.corpus import wordnet as wn
import itertools
from collections import Counter
import re
import copy
import nltk
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
import ssl
import numpy as np
import sklearn

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



class CorpusProcessor:
    def __init__(self, corpus=nltk.corpus.reuters, tagged_lemmatized=True, call_remove_small_words=True,
                 call_remove_stopwords=True, remove_most_frequent=0, frequency_threshold=10,
                 sc_frequency_threshold=10):

        self.un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}

        # Attributes store corpus information at each stage of the pipeline.
        self.corpus = corpus

        self.original_corpus = corpus
        self.c_tagged_lemmatized = None
        self.c_small_words_removed = None
        self.c_stopwords_removed = None
        self.surface_cooccurrences = None

        # Attributes store extra corpus info.
        self.lemma_frequencies = None
        self.filtered_lemma_frequencies = None

        # Instruct behaviour at each stage of the pipeline.
        self.tagged_lemmatized = tagged_lemmatized
        self.call_remove_small_words = call_remove_small_words
        self.call_remove_stopwords = call_remove_stopwords
        self.remove_n_most_frequent = remove_most_frequent
        self.frequency_threshold = frequency_threshold
        self.sc_frequency_threshold = sc_frequency_threshold

        # Run corpus through the pipeline.
        self.process()

    def process(self):
        if self.tagged_lemmatized:
            self.c_tagged_lemmatized = self.tag_lemmatize_corpus()
            self.corpus = copy.deepcopy(self.c_tagged_lemmatized)
        if self.remove_small_words:
            self.c_small_words_removed = self.remove_small_words()
            self.corpus = copy.deepcopy(self.c_small_words_removed)
        if self.remove_stopwords:
            self.c_stopwords_removed = self.remove_stopwords()
            self.corpus = copy.deepcopy(self.c_stopwords_removed)
        self.corpus = self.filter_corpus()
        self.lemma_frequencies = Counter(itertools.chain(*self.corpus))
        self.filtered_lemma_frequencies = Counter(itertools.chain(*self.corpus))
        self.surface_cooccurrences = self.surface_cooccurrences_in_corpus()
        if self.sc_frequency_threshold > 0:
            self.surface_cooccurrences = self.filter_surface_cooccurrences()

    def tag_sentence(self, sentence):
        return nltk.pos_tag(sentence, tagset='universal')

    def lemmatize_tagged_sentence(self, tagged_sentence):
        lemmatized_sentence = []
        for word, position in tagged_sentence:
            if position in [".", "X"]:
                continue
            elif position in self.un2wn_mapping.keys():
                lemma = nltk.WordNetLemmatizer().lemmatize(word, pos=self.un2wn_mapping[position])
            else:
                lemma = nltk.WordNetLemmatizer().lemmatize(word)
            lemmatized_sentence.append("-".join([lemma.lower(), position]))
        return lemmatized_sentence

    # Tag and lemmatize contents of a corpus. Return a list of lists (sentences).
    def tag_lemmatize_corpus(self):
        tagged_lemmatized_corpus = []
        for sentence in self.corpus.sents():
            tagged_sentence = self.tag_sentence(sentence)
            lemmatized_tagged_sentence = self.lemmatize_tagged_sentence(tagged_sentence)
            tagged_lemmatized_corpus.append(lemmatized_tagged_sentence)
        return tagged_lemmatized_corpus

    # Remove words with < 3 alphabetic characters.
    def remove_small_words(self):
        alphabet_regex = re.compile(r'[a-z]', re.IGNORECASE)
        filtered_corpus = []
        for sentence in self.corpus:
            filtered_sentence = []
            for word in sentence:
                if not len(alphabet_regex.findall(word.split('-')[0])) < 3:
                    filtered_sentence.append(word)
            filtered_corpus.append(filtered_sentence)
        return filtered_corpus

    # Remove stopwords from processed corpus.
    def remove_stopwords(self, stopwords=nltk.corpus.stopwords.words('english')):
        filtered_corpus = []
        for sentence in self.corpus:
            filtered_sentence = []
            for word in sentence:
                if word.split('-')[0].lower() not in stopwords:
                    filtered_sentence.append(word)
            filtered_corpus.append(filtered_sentence)
        return filtered_corpus

    # Return counter object of n most frequent words.
    def most_frequent(self, amount):
        lemma_frequencies = Counter(itertools.chain(*self.corpus))
        most_frequent_dict = {}
        for key, value in dict(lemma_frequencies.most_common(amount)).items():
            most_frequent_dict[key] = value
        return most_frequent_dict

    # Return a dict of words below a frequency threshold, and their frequencies.
    def least_frequent(self, frequency_threshold):
        lemma_frequencies = Counter(itertools.chain(*self.corpus))
        least_frequent_dict = {}
        for key, value in dict(reversed(lemma_frequencies.most_common())).items():
            if value <= frequency_threshold:
                least_frequent_dict[key] = value
            else:
                break
        return least_frequent_dict

    # Remove the n highest frequency words, and words below a frequency threshold.
    def filter_corpus(self):
        least_freq = {}
        most_freq = {}
        if self.remove_n_most_frequent > 0:
            least_freq = self.least_frequent(self.frequency_threshold)
        if self.frequency_threshold > 0:
            most_freq = self.most_frequent(self.remove_n_most_frequent)
        words_to_remove_dict = least_freq | most_freq

        filtered_corpus = []
        for index, sentence in enumerate(self.corpus):
            filtered_sentence = []
            for word in sentence:
                if word not in words_to_remove_dict.keys():
                    filtered_sentence.append(word)
                else:
                    words_to_remove_dict[word] -= 1
                    if words_to_remove_dict[word] == 0:
                        del words_to_remove_dict[word]
            filtered_corpus.append(filtered_sentence)
        return filtered_corpus

    def surface_cooccurrences_in_sentence(self, sentence, span):
        word_contexts = []
        for index, word in enumerate(sentence):
            if word.split('-')[-1] in ['NOUN']:
                left_index_range = list(range(max(index - span, 0), index))
                right_index_range = list(range(index + 1, min(index + span, len(sentence))))
                index_range = left_index_range + right_index_range
                for context_word in [sentence[i] for i in index_range]:
                    word_contexts.append((word, context_word))
        return word_contexts

    # Return cooccurrences dictionary from a corpus.
    def surface_cooccurrences_in_corpus(self):
        cooccurrences_counter = Counter()
        for sentence in self.corpus:
            cooccurrences_counter.update(self.surface_cooccurrences_in_sentence(sentence, 5))
        return cooccurrences_counter

    def filter_surface_cooccurrences(self):
        filtered_cooccurrences = {}
        for pair, frequency in self.surface_cooccurrences.items():
            if not frequency < self.sc_frequency_threshold:
                filtered_cooccurrences[pair] = frequency
        return filtered_cooccurrences


class SCReducer:
    def __init__(self, surface_cooccurrences, do_svd=True, do_mds=False, get_distance_matrix=True, svd_dimensions=100,
                 mds_dimensions=2):
        self.is_matrix = False
        self.matrix_labels = None
        self.matrix_labels_no_tags = None
        self.corpus_processor_object = None
        self.surface_cooccurrences = surface_cooccurrences

        if isinstance(surface_cooccurrences, np.ndarray):
            self.is_matrix = True

        if isinstance(surface_cooccurrences, CorpusProcessor):
            self.surface_cooccurrences = surface_cooccurrences.surface_cooccurrences
            self.corpus_processor_object = surface_cooccurrences

        self.svd_dimensions = svd_dimensions
        self.mds_dimensions = mds_dimensions

        self.do_svd = do_svd
        self.do_mds = do_mds
        self.get_distance_matrix = get_distance_matrix

        # Matrices attributes
        self.vector_matrix = None
        self.distance_matrix = None
        self.svd_matrix = None
        self.mds_matrix = None

        self.process()

    def process(self):
        if not self.is_matrix:
            self.sc_to_vector_matrix()
        if self.do_svd:
            self.svd()
        if self.do_mds or self.get_distance_matrix:
            self.generate_distance_matrix()
        if self.do_mds:
            self.mds()

        self.matrix_labels_no_tags = {label.split('-')[0]: index for label, index in self.matrix_labels.items()}

    def sc_to_vector_matrix(self):
        bigram_list = [k for k in self.surface_cooccurrences.keys()]
        target_list, context_list = [i[0] for i in bigram_list], [i[1] for i in bigram_list]
        unique_targets = list(set(target_list))
        unique_contexts = list(set(context_list))
        target_indices = {target: index for index, target in enumerate(unique_targets)}
        context_indices = {context: index for index, context in enumerate(unique_contexts)}
        self.vector_matrix = np.zeros((len(target_indices), len(context_indices)))

        self.matrix_labels = target_indices

        for pair, weight in self.surface_cooccurrences.items():
            self.vector_matrix[target_indices[pair[0]]][context_indices[pair[1]]] = weight

    def svd(self):
        self.svd_matrix = sklearn.decomposition.TruncatedSVD(n_components=self.svd_dimensions,
                                                             algorithm="arpack").fit_transform(self.vector_matrix)

    def generate_distance_matrix(self):
        self.distance_matrix = pairwise_distances(self.vector_matrix, metric="cosine")

    def mds(self):
        mds = MDS(n_components=self.mds_dimensions, max_iter=300, eps=1e-9,
                  random_state=np.random.RandomState(seed=6),
                  dissimilarity="precomputed", normalized_stress='auto')
        self.mds_matrix = mds.fit(self.distance_matrix).embedding_


class CorpusReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ''
        self.sentences = self.read_to_sentences()

    def sents(self):
        return [sentence for sentence in self.sentences]

    # This function opens a file and tokenizes its contents, returning a series of sentences.
    def read_to_sentences(self):
        with open(self.file_path, 'r') as file:
            self.raw_text = file.read()
        tokenized_sentences = []
        sentences = sent_tokenize(self.raw_text)
        for sentence in sentences:
            tokenized_sentences.append(word_tokenize(sentence))
        return tokenized_sentences