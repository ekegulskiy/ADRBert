# ADRBert
Automatic ADR detection using BERT

## Instructions
### Python Requirements
1. make sure you use python3
1. pip3 install gitpython
2. pip3 install -r requirements.txt

### Setup Project Environment
To setup project environments, execute setup_env.py script as following:
```
python setup_env.py
```

It will download and prepare the following sub-directories with components:
1. adrmine_data (ADRMine data from http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip)
2. bert (GitHub repo from https://github.com/google-research/bert.git)
3. bert_generic_model (bert pre-trained model from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)
4. bert_adr_out (bert fine-tuned ADR model)

### Convert ADRMine data to BERT format
1. Use ADRMine script to download training and test set tweets from which ADR annotation were created:

```
python adrmine_data/download_tweets/download_tweets.py \
       adrmine_data/download_tweets/test_tweet_ids.tsv > adrmine_data/download_tweets/test_tweet_posts.tsv

python adrmine_data/download_tweets/download_tweets.py \
       adrmine_data/download_tweets/train_tweet_ids.tsv > adrmine_data/download_tweets/train_tweet_posts.tsv
```

2. Use generate_bert_data.py script to convert ADRMine training and test sets into bert format:

```
python generate_bert_data.py --adrmine-tweets=adrmine_data/download_tweets/test_tweet_posts.tsv \
                       --adrmine-annotations=adrmine_data/download_tweets/test_tweet_annotations.tsv \
                       --json-output-file=adrmine_data/adrmine_test.json

python generate_bert_data.py --adrmine-tweets=adrmine_data/download_tweets/train_tweet_posts.tsv \
                       --adrmine-annotations=adrmine_data/download_tweets/train_tweet_annotations.tsv \
                       --json-output-file=adrmine_data/adrmine_train.json

```

### Fine-tune ADR model
Fine-tuning ADR model involves running bert neural network training and takes about 1 day on a fast Linux PC. That's why
Google compute engine with TPU (Tensorflow Processing Unit) is recommended where it takes around 1 hour. T
he following instructions are using Google TPU:

### Evaluating ADR model
Evaluation of ADR model can be run as on desktop Linux PC as it does not very long time (about 2-4 minutes).
However, it may run out of memory. On 16-GB Linux PC, it exceeded memory usage by 10% but still was able to run.

1. Run bert adr classifier to evaluate test set and create output files required for F1 computation:
```
python bert_adr_classifier.py --vocab_file=bert_generic_model/uncased_L-24_H-1024_A-16/vocab.txt \
                       --bert_config_file=bert_generic_model/uncased_L-24_H-1024_A-16/bert_config.json \
                       --init_checkpoint=bert_generic_model/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                       --do_train=False --train_file=adrmine_data/adrmine_train.json --do_predict=True \
                       --predict_file=adrmine_data/adrmine_test.json --train_batch_size=24 \
                       --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=50 \
                       --output_dir=./bert_adr_model --use_tpu=False --version_2_with_negative=True
```

2. Compute F1 score from bert adr classifier results:
```
python evaluate-v2.0.py adrmine_data/adrmine_test.json bert_adr_model/predictions.json \
                --na-prob-file bert_adr_model/null_odds.json
```