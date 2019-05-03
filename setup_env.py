import os
import urllib
import zipfile
from git import Repo

# ADRMine artifacts
ADRMINE_DATA_URL = "http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip"
ADRMINE_DATA_ZIPFILE = "adrmine_data.zip"
ADRMINE_DATA_DIR = "adrmine_data"

# bert artifacts
BERT_GIT_URL = "https://github.com/google-research/bert.git"
BERT_DIR = "bert"
BERT_LARGE_UNCASED_URL = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
BERT_LARGE_UNCASED_ZIPFILE = "uncased_L-24_H-1024_A-16.zip"
BERT_GENERIC_MODEL_DIR = "bert_generic_model"

# ADRBert artifacts
BERT_ADR_LARGE_URL = "https://storage.cloud.google.com/squad-nn/bert/models/bert_adr_large.zip"
BERT_ADR_LARGE_ZIPFILE = "bert_adr_large.zip"
BERT_ADR_MODEL_DIR = "bert_adr_model"

print("Downloading {}".format(ADRMINE_DATA_URL))
urllib.urlretrieve("{}".format(ADRMINE_DATA_URL), filename=ADRMINE_DATA_ZIPFILE)

print("Unzipping {}".format(ADRMINE_DATA_ZIPFILE))
zip_ref = zipfile.ZipFile(ADRMINE_DATA_ZIPFILE, 'r')
zip_ref.extractall(ADRMINE_DATA_DIR)
zip_ref.close()

if not os.path.exists(BERT_DIR):
    print("Cloning BERT repository from ".format(BERT_GIT_URL))
    Repo.clone_from(BERT_GIT_URL, BERT_DIR)

if not os.path.exists(BERT_LARGE_UNCASED_ZIPFILE):
    print("Downloading Bert base model")
    urllib.urlretrieve("{}".format(BERT_LARGE_UNCASED_URL), filename=BERT_LARGE_UNCASED_ZIPFILE)
    zip_ref = zipfile.ZipFile(BERT_LARGE_UNCASED_ZIPFILE, 'r')
    zip_ref.extractall(BERT_GENERIC_MODEL_DIR)
    zip_ref.close()

if not os.path.exists(BERT_LARGE_UNCASED_ZIPFILE):
    print("Downloading Bert base model")
    urllib.urlretrieve("{}".format(BERT_LARGE_UNCASED_URL), filename=BERT_LARGE_UNCASED_ZIPFILE)
    zip_ref = zipfile.ZipFile(BERT_LARGE_UNCASED_ZIPFILE, 'r')
    zip_ref.extractall(BERT_GENERIC_MODEL_DIR)
    zip_ref.close()

if not os.path.exists(BERT_GENERIC_MODEL_DIR):
    urllib.urlretrieve("{}".format(BERT_LARGE_UNCASED_URL), filename=BERT_LARGE_UNCASED_ZIPFILE)
    zip_ref = zipfile.ZipFile(BERT_LARGE_UNCASED_ZIPFILE, 'r')
    zip_ref.extractall(BERT_GENERIC_MODEL_DIR)
    zip_ref.close()

if not os.path.exists(BERT_ADR_LARGE_URL):
    print("Downloading Bert ADR Large model")
    urllib.urlretrieve("{}".format(BERT_ADR_LARGE_URL), filename=BERT_ADR_LARGE_ZIPFILE)
    zip_ref = zipfile.ZipFile(BERT_ADR_LARGE_ZIPFILE, 'r')
    zip_ref.extractall(BERT_ADR_MODEL_DIR)
    zip_ref.close()