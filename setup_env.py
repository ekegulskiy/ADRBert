import os
import urllib.request
import zipfile
from git import Repo
import progressbar

# ADRMine artifacts
ADRMINE_DATA_URL = "http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip"
ADRMINE_DATA_DIR = "adrmine_data"
ADRMINE_DATA_ADR_LEXICON_URL = "http://diego.asu.edu/downloads/publications/ADRMine/ADR_lexicon.tsv"
ADRMINE_DATA_ADR_LEXICON_NAME = "adrmine_data/ADR_lexicon.tsv"

# bert artifacts
BERT_GIT_URL = "https://github.com/google-research/bert.git"
BERT_DIR = "bert"
BERT_LARGE_UNCASED_URL = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
BERT_GENERIC_MODEL_DIR = "bert_generic_model"

# ADRBert artifacts
BERT_ADR_LARGE_URL = "https://storage.googleapis.com/squad-nn/bert/models/bert_adr_large.zip"
BERT_ADR_MODEL_DIR = "bert_adr_model"


# This function is taken from https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_and_unzip(zip_url, unzip_dir):
    if not os.path.exists(unzip_dir):
        target_zip_file = os.path.basename(zip_url)
        print("Downloading {}".format(zip_url))
        urllib.request.urlretrieve("{}".format(zip_url), target_zip_file, MyProgressBar())
        print("Unzipping {}".format(target_zip_file))
        zip_ref = zipfile.ZipFile(target_zip_file, 'r')
        zip_ref.extractall(unzip_dir)
        zip_ref.close()
        os.remove(target_zip_file)
    else:
        print("{} already exists, skipping download of {}".format(unzip_dir, zip_url))


if not os.path.exists(BERT_DIR):
    print("Cloning BERT repository from ".format(BERT_GIT_URL))
    Repo.clone_from(BERT_GIT_URL, BERT_DIR)

download_and_unzip(ADRMINE_DATA_URL, ADRMINE_DATA_DIR)
download_and_unzip(BERT_LARGE_UNCASED_URL, BERT_GENERIC_MODEL_DIR)
download_and_unzip(BERT_ADR_LARGE_URL, BERT_ADR_MODEL_DIR)

print("Downloading {}".format(ADRMINE_DATA_ADR_LEXICON_NAME))
urllib.request.urlretrieve("{}".format(ADRMINE_DATA_ADR_LEXICON_URL), ADRMINE_DATA_ADR_LEXICON_NAME)
