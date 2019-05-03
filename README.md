# ADRBert
Automatic ADR detection using BERT


bert_classifier --vocab_file=/home/ek/working/sfsu/bert_models/uncased_L-24_H-1024_A-16/vocab.txt \
                --bert_config_file=/home/ek/working/sfsu/bert_models/uncased_L-24_H-1024_A-16/bert_config.json \
                --init_checkpoint=/home/ek/working/sfsu/bert_models/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                --do_train=False --train_file=/home/ek/working/sfsu/ADRBert/adrmine_train.json --do_predict=True \
                --predict_file=/home/ek/working/sfsu/ADRBert/adrmine_test.json --train_batch_size=24 \
                --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=50 \
                --output_dir=./bert_adr_model --use_tpu=False --version_2_with_negative=True


## Instructions
### Python Requirements
1. pip install gitpython
2. pip install -r requirements.txt

### Setup Project Environment
To setup project environments, execute setup_env.py script. It will download and prepare the following project components:
1. ADRMine data (http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip)

### Convert ADRMine data to BERT format
1. Obtain ADRMine data from 
1. Use generate_bert_data.py script. It 2 needs 2 arguments to specify ADRMine tweets file and ADRMine annotations file,
e.g.

```
python generate_bert_data.py --adrmine-tweets adrmine_data/test_tweet.out \
                             --adrmine-annotations adr_mine/test_tweet_annotations.tsv \
                             --json-output-file adrmine_test.json
```