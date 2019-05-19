"""
This file implements the ADR SVM Classifier

Package: adr_svm_classifier

Author: Eduard Kegulskiy

Credits Declaimer: this file uses code snippets from few public sites, the credits are given in each place where used below.

"""

import adrmine_data_loader
import argparse
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from sklearn import model_selection, svm

# =========================== CONSTANTS ==============================
ADR_MENTION_CLASS_NAME = "ADR_MENTION"
NON_ADR_MENTION_CLASS_NAME = "NON_ADR_MENTION"

ADR_MENTION_CLASS_LABEL = 1
NON_ADR_MENTION_CLASS_LABEL = -1

TRAINING_VALIDATION_SPLIT_RATIO = 0.3
# ====================================================================


def load_adr_lexicon(ard_lexicon_file):
    """loads ADR Lexicon from provided file into dict

    # Arguments
        ard_lexicon_file - path to ADR Lexicon file

    # Returns
        dict with ADR Lexicon entries
    """

    print("Loading ADRMine Lexicon from {}...".format(ard_lexicon_file))

    adr_lexicon_dict = []
    with open(ard_lexicon_file) as f:
        for line in f:
            # Each line contains the UMLS (Unified Medical Language System) concept ID,
            # concept name and the source that the concept and the ID are taken from (tab separated).
            # e.g. c1328274	infection vascular	SIDER
            try:
                (UMLS_id, concept_name, source) = line.rstrip().split('\t')
                #print("{}, {}, {}".format(UMLS_id, concept_name, source))
                adr_lexicon_dict.append(concept_name)
            except:
                #print("Ignoring line: {}".format(line))
                pass

    print("    {} entries loaded".format(len(adr_lexicon_dict)))
    return adr_lexicon_dict


def is_in_adr_lexicon(text, adr_lexicon_dict):
    """checks if given text is present in ADR Lexicon dict

    # Arguments
        text - text to check
        adr_lexicon_dict - dict with ADR Lexicon entries

    # Returns
        True if present, False otherwise
    """
    for item in adr_lexicon_dict:
        if item.lower() == text.lower():
            return True

    return False


def check_adr_lexicon(annotations_dict, adr_lexicon_dict):
    """prints statistics ADR Mentions and ADR Indications related to ADR Lexicon
       i.e. counts how many are present in ADR Lexicon vs. not

    # Arguments
        annotations_dict - dict with annotations
        adr_lexicon_dict - dict with ADR Lexicon entries

    # Returns
        None
    """

    adrs_matching_labels = 0
    adrs_not_found_in_lexicon = 0
    indications_matching_labels = 0
    indications_not_found_in_lexicon = 0
    for i, (k, v) in enumerate(annotations_dict.items()):
        for index, annotation in enumerate(v):
            # tweet = tweets_dict[k]
            annotatedText = annotation['annotatedText']

            is_adr_lexicon = is_in_adr_lexicon(annotatedText, adr_lexicon_dict)
            if is_adr_lexicon:
                # print("ADR lexicon contains this text {}".format(annotatedText))
                # detected_adrs += 1
                if annotation['semanticType'] == "ADR":
                    adrs_matching_labels += 1
                else:
                    indications_matching_labels += 1
            else:
                if annotation['semanticType'] == "ADR":
                    adrs_not_found_in_lexicon += 1
                else:
                    indications_not_found_in_lexicon += 1

    print("Number of ADR mentions present in the ADR Lexicon: {}".format(adrs_matching_labels))
    print("Number of Indication mentions present in the ADR Lexicon: {}".format(indications_matching_labels))
    print("Number of ADR mentions not present in the ADR Lexicon: {}".format(adrs_not_found_in_lexicon))
    print("Number of Indication mentions not present in the ADR Lexicon: {}".format(indications_not_found_in_lexicon))


def vectorize_vocabulary(train_tweets_dict, test_tweets_dict):
    """vectorizes all text corpus from the train and test sets

    # Arguments
        train_tweets_dict - dict with entries from training set tweets
        test_tweets_dict - dict with entries from test set tweets

    # Returns
        TfidfVectorizer object
    """

    print("Vectorizing ADRMine data vocabulary...")

    tfidf_vectorizer = TfidfVectorizer()
    corpus = []

    for i, (k, v) in enumerate(train_tweets_dict.items()):
        corpus.append(v.lower())

    for i, (k, v) in enumerate(test_tweets_dict.items()):
        corpus.append(v.lower())

    tfidf_vectorizer.fit_transform(corpus)
    #print(Tfidf_vect.vocabulary_)
    #print(len(Tfidf_vect.vocabulary_))
    #print(Tfidf_vect.idf_)
    print("    size of vocabulary: {}".format(len(tfidf_vectorizer.vocabulary_)))
    return tfidf_vectorizer


def balance_set(X, Y, adr_labels_size, nonadr_labels_size):
    """balances the set by doing up- and down -sampling to converge into the same class size

    # Arguments
        X - set samples
        Y - set labels
        adr_labels_size - ADR_MENTION_CLASS size
        nonadr_labels_size - NON_ADR_MENTION_CLASS size

    # Returns
        new_X - new balanced samples
        new_Y - new labels corresponding to new_X
    """

    print("Performing Class Balancing...")
    adr_samples_needed = nonadr_labels_size - adr_labels_size
    new_X = []
    new_Y = []
    adr_labels_size = 0
    nonadr_labels_size = 0

    for index, example in enumerate(X):
        if adr_samples_needed > 0:
            if Y[index] == ADR_MENTION_CLASS_LABEL:
                new_X.append(example)  # add original 'ADR' sample
                new_Y.append(ADR_MENTION_CLASS_LABEL)
                new_X.append(example)  # add duplicate 'ADR' sample to perform Over-Sampling
                new_Y.append(ADR_MENTION_CLASS_LABEL)

                adr_labels_size += 2
                adr_samples_needed -= 1
            else:
                # we don't add original 'No ADR Mention' sample to perform Under-Sampling
                adr_samples_needed -= 1

        else:
            if Y[index] == ADR_MENTION_CLASS_LABEL:
                adr_labels_size += 1
            else:
                nonadr_labels_size += 1

            new_X.append(example)  # add original sample
            new_Y.append(Y[index])  # add original label

    print("    Updated dataset size: {}".format(len(new_X)))
    print("    {} class size: {}".format(ADR_MENTION_CLASS_NAME, adr_labels_size))
    print("    {} class size: {}".format(NON_ADR_MENTION_CLASS_NAME, nonadr_labels_size))

    return new_X, new_Y

def build_data_vectors(annotations, tweets, Tfidf_vect, adr_lexicon_dict, should_balance_set=True):
    """builds training and test data vectors from the annotated tweets using feature extraction

    # Arguments
        annotations - dict with annotations of the dataset
        tweets - dict with tweets of the dataset
        Tfidf_vect - TfidfVectorizer object fitted with the Corpus
        adr_lexicon_dict - dict with ADR Lexicon entries
        should_balance_set - whether to balance the set

    # Returns
        X - vectorized samples in the csr_matrix format as expected by SVM classifier
        Y - list of class labels corresponding to entries in X
    """

    def vectorize_word(word):
        """gives vectorized value from TfidfVectorizer for the given word
           If the word is not part of vocabulary, 0 will be returned

        # Arguments
            word - word to vectorize

        # Returns
            vectorized value
        """
        if word in Tfidf_vect.vocabulary_:
            i = Tfidf_vect.vocabulary_[word]
            return Tfidf_vect.idf_[i]
        else:
            return 0

    def clean_text(text):
        """Cleans the text
           This code snippet is taken from https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
           Author: Susan Li

        # Arguments
            text - text to clean

        # Returns
            cleaned text
        """
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text

    X = []
    Y = []
    adr_labels_size = 0
    nonadr_labels_size = 0
    for i, (k, v) in enumerate(annotations.items()):
        tweet_text = clean_text(tweets[k])
        tokens = word_tokenize(tweet_text)

        for annotation_index, annotation in enumerate(v):
            prev_token_adr = False

            annotated_text = clean_text(annotation['annotatedText'])
            annotated_text_tokens = word_tokenize(annotated_text)

            for index, focus_word in enumerate(tokens):
                focus_vector = []

                # for Context feature, get index for 3 surrounding words on each side of focus word
                if program_args.context_feature:
                    focus_vector.append(vectorize_word(tokens[index-3]) if (index-3 >= 0) else 0)
                    focus_vector.append(vectorize_word(tokens[index-2]) if (index-2 >= 0) else 0)
                    focus_vector.append(vectorize_word(tokens[index-1]) if (index-1 >= 0) else 0)
                    focus_vector.append(vectorize_word(tokens[index]))
                    focus_vector.append(vectorize_word(tokens[index+1]) if (index+1 < len(tokens)) else 0)
                    focus_vector.append(vectorize_word(tokens[index+2]) if (index+2 < len(tokens)) else 0)
                    focus_vector.append(vectorize_word(tokens[index+3]) if (index+3 < len(tokens)) else 0)

                if program_args.adrlexicon_feature:
                    if focus_word in adr_lexicon_dict:
                        focus_vector.append(1)
                    else:
                        focus_vector.append(0)

                if program_args.prev_adrlexicon_feature:
                    if prev_token_adr:
                        focus_vector.append(1)
                    else:
                        focus_vector.append(0)

                # assign class label
                if annotation['semanticType'] == 'ADR' and focus_word in annotated_text_tokens:
                    Y.append(ADR_MENTION_CLASS_LABEL)
                    X.append(focus_vector)
                    adr_labels_size += 1
                    prev_token_adr = True
                else:
                    Y.append(NON_ADR_MENTION_CLASS_LABEL)
                    X.append(focus_vector)
                    nonadr_labels_size += 1
                    prev_token_adr = False

    print("    Dataset size: {}".format(len(X)))
    print("    {} class size: {}".format(ADR_MENTION_CLASS_NAME, adr_labels_size))
    print("    {} class size: {}".format(NON_ADR_MENTION_CLASS_NAME, nonadr_labels_size))

    if should_balance_set:
        X, Y = balance_set(X, Y, adr_labels_size, nonadr_labels_size)

    X = scipy.sparse.csr_matrix(X)
    return X, Y


def calf_f1(annotated_Y, predicted_Y):
    """Calculates F1, Precision and Recall for the given pair of annotated and predicted class labels and prints them out

    # Arguments
        annotated_Y - list of annotated labels
        predicted_Y - list of predicted labels

    # Returns
        None
    """

    POSITIVE = ADR_MENTION_CLASS_LABEL
    NEGATIVE = NON_ADR_MENTION_CLASS_LABEL

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    total_actual_positives = 0
    total_actual_negatives = 0

    for index, actual in enumerate(annotated_Y):
        predicted = predicted_Y[index]

        if actual == POSITIVE:
            total_actual_positives += 1

            if predicted == POSITIVE:
                tp += 1
            elif predicted == NEGATIVE:
                fn += 1

        elif actual == NEGATIVE:
            total_actual_negatives += 1

            if predicted == POSITIVE:
                fp += 1
            elif predicted == NEGATIVE:
                tn += 1

    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)

    if (tp+fn) == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)

    if (precision+recall) == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)

    # print("Total labels: {}, total actual positives: {}, total_actual_negatives: {}".format(len(predicted_Y), total_actual_positives, total_actual_negatives))
    # print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp, tn, fp, fn))
    # print("        Accuracy: {}".format((tp+tn)/(len(test_Y))))
    print("        Precision: {}".format(precision))
    print("        Recall: {}".format(recall))
    print("        F1: {}".format(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datasets args
    parser.add_argument('--train-adrmine-tweets', required=True, type=str, help='ADRMine training dataset file with tweets')
    parser.add_argument('--train-adrmine-annotations', required=True, type=str, help='ADRMine training dataset file with annotations')
    parser.add_argument('--test-adrmine-tweets', required=True, type=str, help='ADRMine test dataset file with tweets')
    parser.add_argument('--test-adrmine-annotations', required=True, type=str, help='ADRMine test dataset file with annotations')
    parser.add_argument('--adrmine-adr-lexicon', required=True, type=str, help='ADRMine ADR Lexicon file')

    # features args
    parser.add_argument('--context-feature', dest='context_feature', action='store_true')
    parser.add_argument('--no-context-feature', dest='context_feature', action='store_false')
    parser.set_defaults(context_feature=True)
    parser.add_argument('--adrlexicon-feature', dest='adrlexicon_feature', action='store_true')
    parser.add_argument('--no-adrlexicon-feature', dest='adrlexicon_feature', action='store_false')
    parser.set_defaults(adrlexicon_feature=True)
    parser.add_argument('--prev-adrlexicon-feature', dest='adrlexicon_feature', action='store_true')
    parser.add_argument('--no-prev-adrlexicon-feature', dest='prev_adrlexicon_feature', action='store_false')
    parser.set_defaults(prev_adrlexicon_feature=True)

    program_args = parser.parse_args()

    admine_training_data = adrmine_data_loader.ADRMineDataLoader()
    admine_test_data = adrmine_data_loader.ADRMineDataLoader()

    (train_annotations, train_tweets) = admine_training_data.load(program_args.train_adrmine_tweets,
                                                                  program_args.train_adrmine_annotations)
    (test_annotations, test_tweets) = admine_test_data.load(program_args.test_adrmine_tweets,
                                                            program_args.test_adrmine_annotations)

    adr_lexicon = load_adr_lexicon(program_args.adrmine_adr_lexicon)

    # uncomment this function to test how many annotatated ADR Mentions are present in ADR Lexicon.
    # check_adr_lexicon(train_annotations, adr_lexicon);

    Tfidf_vect = vectorize_vocabulary(train_tweets, test_tweets)
    print("Building feature vectors for training...")
    (train_X, train_Y) = build_data_vectors(train_annotations, train_tweets, Tfidf_vect, adr_lexicon, should_balance_set=True)
    print("Building feature vectors for testing...")
    (test_X, test_Y) = build_data_vectors(test_annotations, test_tweets, Tfidf_vect, adr_lexicon, should_balance_set=False)

    # split the training set to do validation
    train_X, Valid_X, train_Y, Valid_Y = model_selection.train_test_split(train_X, train_Y, random_state = 100,
                                                                          test_size=TRAINING_VALIDATION_SPLIT_RATIO)

    # Run SVM Classifier
    # The SVM init and fitting code is taken from https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34)
    # denoted by CODE_SNIPPET_BEGIN and CODE_SNIPPET_END

    # -------------------- CODE_SNIPPET_BEGIN -----------------------
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    print("Running SVM Classifier...")
    print("    Training...")
    SVM.fit(train_X, train_Y)
    # predict the labels on validation dataset
    print("    Evaluating Validation Set...")
    predictions_SVM = SVM.predict(Valid_X)
    # -------------------- CODE_SNIPPET_END -------------------------

    # Use accuracy_score function to get the accuracy
    calf_f1(Valid_Y, predictions_SVM)

    predictions_SVM = SVM.predict(test_X)
    # Use accuracy_score function to get the accuracy
    print("    Evaluating Test Set...")
    calf_f1(test_Y, predictions_SVM)