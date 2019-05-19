"""
This file implements conversion from ADRMine data to BERT SQuAD JSON format

Package: generate_bert_data

Author: Eduard Kegulskiy

"""

# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import argparse
import adrmine_data_loader


def convert_to_json(annotations_dict, tweets_dict, output_file):
    """converts ADRMine data into JSON format as specified by https://rajpurkar.github.io/SQuAD-explorer/

    # Arguments
        annotations_dict - dict with ADRMine annotations
        tweets_dict - dict with ADRMine tweets
        output_file - JSON file to be created

    # Returns
        None
    """

    def contains_adr(annotation_list):
        # first check whether there is at least one ADR mention in this tweet
        for index, annotation in enumerate(annotation_list):
            if annotation['semanticType'] == "ADR":
                return True

        return False

    num_adr_samples = 0
    num_no_adr_samples = 0

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
                num_adr_samples += 1
            else:
                if does_contain_adr is False:
                    # we only add empty answers when the tweet does not contain an ADR, otherwise we just skip it
                    data['data'][0]['paragraphs'][i]['qas'].append(None)
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries] = {}
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['question'] = "Is ADR mentioned?"
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['id'] = "{}-{}".format(k, num_qas_entries)

                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['answers'] = []
                    data['data'][0]['paragraphs'][i]['qas'][num_qas_entries]['is_impossible'] = True

                    num_qas_entries += 1
                    num_no_adr_samples += 1

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile)

    print("Converted JSON Data:")
    print("     Number of samples containing ADR mentions: {}".format(num_adr_samples))
    print("     Number of samples without ADR mentions: {}".format(num_no_adr_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adrmine-tweets', required=True, type=str, help='ADRMine dataset file with tweets')
    parser.add_argument('--adrmine-annotations', required=True, type=str, help='ADRMine dataset file with annotations')
    parser.add_argument('--json-output-file', required=True, type=str, help='output file in JSON format')

    program_args = parser.parse_args()

    admine_data_loader = adrmine_data_loader.ADRMineDataLoader()
    (annotationsDict, tweetTextDict) = admine_data_loader.load(program_args.adrmine_tweets, program_args.adrmine_annotations)

    convert_to_json(annotationsDict, tweetTextDict, program_args.json_output_file)