"""

"""

# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import argparse
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re, string

class ADRMineDataLoader:

    def __init__(self):
        self._adrmine_tweets = None
        self._adrmine_annotations = None
        self._annotations_dict = None
        self._tweets_dict = None

    def _validate_annotations(self):
        for i, (k, v) in enumerate(self._annotations_dict.items()):
            for index, annotation in enumerate(v):
                startOffset = int(annotation['startOffset'])
                endOffset = int(annotation['endOffset'])
                tweet = self._tweets_dict[k]
                annotatedText = annotation['annotatedText']

                realOffset = tweet.find(annotatedText)
                if realOffset != startOffset:
                    print("Fixing startOffset for {}. (annotated at position {}, but should be at {})".format(k, startOffset, realOffset))

                    diff = realOffset - startOffset
                    annotation['startOffset'] = "{}".format(startOffset+diff)
                    annotation['endOffset'] = "{}".format(endOffset+diff)


    def load(self, adrmine_tweets, adrmine_annotations):
        self._adrmine_tweets = adrmine_tweets
        self._adrmine_annotations = adrmine_annotations

        num_missing_tweets = 0
        self._tweets_dict = {}
        with open(self._adrmine_tweets) as f:
            for line in f:
                # each line contains 4 fields, tab-separated:
                # tweet ID, user ID, text ID and Tweet text
                (tweetID, userID, textID, tweetText) = line.rstrip().split('\t')
                self._tweets_dict[textID] = tweetText

        self._annotations_dict = {}
        adrmine_orig_annotations = 0
        num_usable_annotations = 0
        with open(self._adrmine_annotations) as f:
            for line in f:
                # each line contains 5 fields, tab-separated:
                # text ID, start offset, end offset, semantic type, annotated text, related drug and target drug.
                (textID, startOffset, endOffset, semanticType, annotatedText, relatedDrug, targetDrug) = line.rstrip().split('\t')

                if textID in self._tweets_dict:
                    if textID not in self._annotations_dict:
                        self._annotations_dict[textID] = []

                    self._annotations_dict[textID].append({'semanticType': semanticType,
                                                'startOffset': startOffset,
                                                'endOffset': endOffset,
                                                'annotatedText': annotatedText})
                    num_usable_annotations += 1
                else:
                    print("TextID {} does not have a corresponding tweet".format(textID))
                    num_missing_tweets += 1

                adrmine_orig_annotations += 1

        self._validate_annotations()

        print("Original ADRMine Data:")
        print("    Number of original annotations: {}".format(adrmine_orig_annotations))
        print("    Number of missing tweets: {}".format(num_missing_tweets))
        print("    Number of usable annotations: {}".format(num_usable_annotations))

        return (self._annotations_dict, self._tweets_dict)