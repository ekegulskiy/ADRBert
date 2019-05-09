import adrmine_data_loader
import argparse

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy
from sklearn.metrics import f1_score

def load_adr_lexicon(ard_lexicon_file):
    print("Loading ADRMine Lexicon from {}...".format(ard_lexicon_file))

    adr_lexicon = []
    with open(ard_lexicon_file) as f:
        for line in f:
            # Each line contains the UMLS (Unified Medical Language System) concept ID,
            # concept name and the source that the concept and the ID are taken from (tab separated).
            # e.g. c1328274	infection vascular	SIDER
            try:
                (UMLS_id, concept_name, source) = line.rstrip().split('\t')
                #print("{}, {}, {}".format(UMLS_id, concept_name, source))
                adr_lexicon.append(concept_name)
            except:
                #print("Ignoring line: {}".format(line))
                pass

    print("    {} entries loaded".format(len(adr_lexicon)))
    return adr_lexicon

def is_in_adr_lexicon(text, adr_lexicon):
    for item in adr_lexicon:
        if item.lower() == text.lower():
            return True

    return False

def check_adr_lexicon(annotations_dict, adr_lexicon):
    adrs_matching_labels = 0
    adrs_not_found_in_lexicon = 0
    indications_matching_labels = 0
    indications_not_found_in_lexicon = 0
    for i, (k, v) in enumerate(annotations_dict.items()):
        for index, annotation in enumerate(v):
            # tweet = tweets_dict[k]
            annotatedText = annotation['annotatedText']

            is_adr_lexicon = is_in_adr_lexicon(annotatedText, adr_lexicon)
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

def vectorize_vocabulary(train_tweets, test_tweets):
    print("Vectorizing ADRMine data vocabulary...")

    Tfidf_vect = TfidfVectorizer()

    corpus = []
    for i, (k, v) in enumerate(train_tweets.items()):
        corpus.append(v.lower())

    for i, (k, v) in enumerate(test_tweets.items()):
        corpus.append(v.lower())

    Tfidf_vect.fit_transform(corpus)
    #print(Tfidf_vect.vocabulary_)
    #print(len(Tfidf_vect.vocabulary_))
    #print(Tfidf_vect.idf_)
    print("    size of vocabulary: {}".format(len(Tfidf_vect.vocabulary_)))
    return Tfidf_vect

def build_data_vectors(annotations, tweets, Tfidf_vect, adr_lexicon, balance_set=True):
    def word_numberic_value(word):
        if word in Tfidf_vect.vocabulary_:
            index = Tfidf_vect.vocabulary_[word]
            return Tfidf_vect.idf_[index]
        else:
            return 0

    CLASS_SIZE_DIFFERENCE_THREASHOLD = 20
    # key doesn't exist in dict

    X = []
    Y = []
    adr_labels_size = 0
    nonadr_labels_size = 0
    for i, (k, v) in enumerate(annotations.items()):
        tweet_text = tweets[k]

        for index, annotation in enumerate(v):
            tokens = word_tokenize(tweet_text.lower())
            annotated_text = annotation['annotatedText'].lower()
            annotated_text_tokens = word_tokenize(annotated_text)

            for index, focus_word in enumerate(tokens):
                focus_vector = []
                # get index for 3 surrounding words on each side of focus word
                if program_args.context_feature:
                    focus_vector.append(word_numberic_value(tokens[index-3]) if (index-3 >= 0) else 0)
                    focus_vector.append(word_numberic_value(tokens[index-2]) if (index-2 >= 0) else 0)
                    focus_vector.append(word_numberic_value(tokens[index-1]) if (index-1 >= 0) else 0)
                    focus_vector.append(word_numberic_value(tokens[index]))
                    focus_vector.append(word_numberic_value(tokens[index+1]) if (index+1 < len(tokens)) else 0)
                    focus_vector.append(word_numberic_value(tokens[index+2]) if (index+2 < len(tokens)) else 0)
                    focus_vector.append(word_numberic_value(tokens[index+3]) if (index+3 < len(tokens)) else 0)

                if program_args.adrlexicon_feature:
                    focus_vector.append(1 if focus_word in adr_lexicon else 0)

                # create label
                if annotation['semanticType'] == 'ADR' and focus_word in annotated_text_tokens:
                    Y.append(1)
                    adr_labels_size += 1
                else:
                    if balance_set and nonadr_labels_size - adr_labels_size > CLASS_SIZE_DIFFERENCE_THREASHOLD:
                        continue

                    Y.append(-1)
                    nonadr_labels_size += 1

                X.append(focus_vector)

    print("    dataset size: {}".format(len(X)))
    print("    'ADR' class size: {}".format(adr_labels_size))
    print("    'NON-ADR' class size: {}".format(nonadr_labels_size))
    X = scipy.sparse.csr_matrix(X)
    return X, Y

if __name__ == '__main__':
    num_missing_tweets = 0

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


    program_args = parser.parse_args()

    admine_training_data = adrmine_data_loader.ADRMineDataLoader()
    admine_test_data = adrmine_data_loader.ADRMineDataLoader()

    (train_annotations, train_tweets) = admine_training_data.load(program_args.train_adrmine_tweets, program_args.train_adrmine_annotations)
    (test_annotations, test_tweets) = admine_test_data.load(program_args.test_adrmine_tweets, program_args.test_adrmine_annotations)

    adr_lexicon = load_adr_lexicon(program_args.adrmine_adr_lexicon)

    Tfidf_vect = vectorize_vocabulary(train_tweets, test_tweets)
    print("Building feature vectors for training...")
    (train_X, train_Y) = build_data_vectors(train_annotations, train_tweets, Tfidf_vect, adr_lexicon)
    print("Building feature vectors for testing...")
    (test_X, test_Y) = build_data_vectors(test_annotations, test_tweets, Tfidf_vect, adr_lexicon, balance_set=False)

    # Run SVM Classifier
    # (code below is using snippets from https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    print("Running SVM Classifier...")
    print("    Training...")
    SVM.fit(train_X, train_Y)
    # predict the labels on validation dataset
    print("    Predicting...")
    predictions_SVM = SVM.predict(test_X)
    # Use accuracy_score function to get the accuracy
    print("    Accuracy Score: {}".format(accuracy_score(predictions_SVM, test_Y) * 100))
    print("    F1 score: {}".format(f1_score(test_Y, predictions_SVM, average='macro')))