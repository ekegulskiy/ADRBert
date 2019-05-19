"""
This file implements the ADRMineDataLoader class

Package: adr_data_loader

Author: Eduard Kegulskiy

"""

# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class ADRMineDataLoader:
    """ Class for loading ADRMine data into corresponding annotations and tweets, interconnected by Unique_ID
        (see http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip)

               # Arguments
                   None

               # Returns
                   A ADRMineDataLoader instance.

               # Examples

               ```python
               import adrmine_data_loader

               # create training and test instances
               admine_training_data = adrmine_data_loader.ADRMineDataLoader()
               admine_test_data = adrmine_data_loader.ADRMineDataLoader()

               # load ADRMine train annotations and tweet text from ADRMine corresponding files
               (train_annotations, train_tweets) = admine_training_data.load(train_adrmine_tweets,
                                                                             train_adrmine_annotations)

               # load ADRMine test annotations and tweet text from ADRMine corresponding files
               (test_annotations, test_tweets) = admine_test_data.load(test_adrmine_tweets,
                                                                       test_adrmine_annotations)


               ```
       """
    def __init__(self):
        self._adrmine_tweets = None
        self._adrmine_annotations = None
        self._annotations_dict = None
        self._tweets_dict = None

    def _validate_annotations(self):
        """checks original ADRMine annotations and fixes the offsets when they are incorrect

        # Arguments
            None

        # Returns
            None
        """
        for i, (k, v) in enumerate(self._annotations_dict.items()):
            for index, annotation in enumerate(v):
                startOffset = int(annotation['startOffset'])
                endOffset = int(annotation['endOffset'])
                tweet = self._tweets_dict[k]
                annotatedText = annotation['annotatedText']

                realOffset = tweet.find(annotatedText)
                if realOffset != startOffset:
                    #print("Fixing startOffset for {}. (annotated at position {}, but should be at {})".format(k, startOffset, realOffset))

                    diff = realOffset - startOffset
                    annotation['startOffset'] = "{}".format(startOffset+diff)
                    annotation['endOffset'] = "{}".format(endOffset+diff)

    def load(self, adrmine_tweets, adrmine_annotations):
        """loads ADRMine data into corresponding annotations and tweets, interconnected by Unique_ID
           ADRMine data files are provided as part of http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip

        # Arguments
            adrmine_tweets - file containing original ADRMine tweets
            adrmine_annotations - file containing original ADRMine annotations

        # Returns
            annotations and tweets dicts
        """

        print("Loading ADRMine data from {}...".format(adrmine_annotations))

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
                    #print("TextID {} does not have a corresponding tweet".format(textID))
                    num_missing_tweets += 1

                adrmine_orig_annotations += 1

        self._validate_annotations()

        print("    Number of original annotations: {}".format(adrmine_orig_annotations))
        print("    Number of missing tweets: {}".format(num_missing_tweets))
        print("    Number of usable annotations: {}".format(num_usable_annotations))

        return (self._annotations_dict, self._tweets_dict)