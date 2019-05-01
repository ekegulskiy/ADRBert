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


def validate_annotations(annotations_dict, tweets_dict):
    for i, (k, v) in enumerate(annotations_dict.items()):
        for index, annotation in enumerate(v):
            startOffset = int(annotation['startOffset'])
            endOffset = int(annotation['endOffset'])
            tweet = tweets_dict[k]
            annotatedText = annotation['annotatedText']

            realOffset = tweet.find(annotatedText)
            if realOffset != startOffset:
                print("startOffset is wrong for {}. Annotated as {}, but should be {}".format(k, startOffset, realOffset))

                diff = realOffset - startOffset
                annotation['startOffset'] = "{}".format(startOffset+diff)
                annotation['endOffset'] = "{}".format(endOffset+diff)


def convert_to_json(annotations_dict, tweets_dict, output_file):

    def contains_adr(annotation_list):
    # first check whether there is at least one ADR mention in this tweet
        for index, annotation in enumerate(annotation_list):
            if annotation['semanticType'] == "ADR":
                return True

        return False

    data = {}
    data['version'] = "v2.0"
    data['data'] = [None]
    data['data'][0] = {}
    data['data'][0]['title'] = "Title"
    data['data'][0]['paragraphs'] = []

    for i, (k, v) in enumerate(annotations_dict.items()):
        data['data'][0]['paragraphs'].append(None)
        data['data'][0]['paragraphs'][i] = {}
        data['data'][0]['paragraphs'][i]['context'] = tweets_dict[k]
        data['data'][0]['paragraphs'][i]['qas'] = []


        if k == "effexor-51d7949453785f584a9b13eb":
            print(k)

        does_contain_adr = contains_adr(v)

        num_qas_entries = 0
        for index, annotation in enumerate(v):
            if annotation['semanticType'] == "ADR":
                data['data'][0]['paragraphs'][i]['qas'].append(None)
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries] = {}
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['question'] = "Is ADR mentioned?"
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['id'] = "{}-{}".format(k, index)

                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'] = [None]
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'][0] = {}
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'][0]['text'] = annotation['annotatedText']
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'][0]['answer_start'] = int(annotation['startOffset'])
                data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['is_impossible'] = False

                num_qas_entries += 1
            else:
                if does_contain_adr is False:
                    # we only add empty answers when the tweet does not contain an ADR, otherwise we just skip it
                    data['data'][0]['paragraphs'][i]['qas'].append(None)
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries] = {}
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['question'] = "Is ADR mentioned?"
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['id'] = "{}-{}".format(k, num_qas_entries)

                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'] = "[]"
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['is_impossible'] = True

                    num_qas_entries += 1

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    num_missing_tweets = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets-text', required=True, type=str, help='ADRMine dataset file with tweets')
    parser.add_argument('--tweets-annotations', required=True, type=str, help='ADRMine dataset file with annotations')
    parser.add_argument('--json-output-file', required=True, type=str, help='output file in JSON format')

    #parser.add_argument('--train_ratio', default=1., type=float,
    #                    help='ratio for train/val split')
    #parser.add_argument('--remove_stopwords', dest='remove_stopwords', action='store_true')
    #parser.add_argument('--use_stemming', dest='use_stemming', action='store_true')
    #parser.add_argument('--clean_text', dest='clean_text', action='store_true')

    program_args = parser.parse_args()

    tweetTextDict = {}
    with open(program_args.tweets_text) as f:
        for line in f:
            # each line contains 4 fields, tab-separated:
            # tweet ID, user ID, text ID and Tweet text
            (tweetID, userID, textID, tweetText) = line.rstrip().split('\t')
            tweetTextDict[textID] = tweetText

    annotationsDict = {}
    with open(program_args.tweets_annotations) as f:
        for line in f:
            # each line contains 5 fields, tab-separated:
            # text ID, start offset, end offset, semantic type, annotated text, related drug and target drug.
            (textID, startOffset, endOffset, semanticType, annotatedText, relatedDrug, targetDrug) = line.rstrip().split('\t')

            if textID in tweetTextDict:
                if textID not in annotationsDict:
                    annotationsDict[textID] = []

                annotationsDict[textID].append({'semanticType': semanticType,
                                            'startOffset': startOffset,
                                            'endOffset': endOffset,
                                            'annotatedText': annotatedText})
            else:
                print("TextID {} does not have a corresponding tweet".format(textID))
                num_missing_tweets += 1


    print("Num of annotations missing tweets is {}, with tweets is {}".format(num_missing_tweets,len(annotationsDict)))

    validate_annotations(annotationsDict,tweetTextDict)
    convert_to_json(annotationsDict, tweetTextDict, program_args.json_output_file)
