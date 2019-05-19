# ADRBert
## Overview
### Objective
 Social Media has undoubtfully become one of the major sources of information over the last decade. Much research has been done to study how users’ posts can help monitor Adverse Drug Reaction (ADR) cases and what can be done to prevent them. One of such research projects developed a system called ADRMine [3], producing a state-of-the-art ADR detection system with F-score of 0.82. The system is based on a machine learning method called Conditional Random Fields (CRF), belonging to the “sequence modeling family” of “probabilistic graphical models” [7]. The objective of my project is to apply a deep learning technique to ADR detection and compare to classical machine learning approaches such as CRF and SVM.
### Methods
 There are 2 methods I use in the project to compare deep learning ADR detection approach to the classical ones. First, a deep learning classifier is developed, ADRBert, using BERT system [1] from recent Google research. BERT is a neural network system based on learning and encoding general language representation model from very large text corpus. First, I fine-tune BERT’s model with ADRMine training data and evaluate the performance with ADRMine test data. Second, I develop simple SVM Classifier, ADR SVM, to serve as a baseline representing classical machine learning approaches. The ADR SVM is using 3 manually engineered features – a sub-set of features used in ADRMine system.
### Results
 ADRBert yields a decent F1 score of 0.701 which is just slightly lower than ADRMine F1 score of 0.721. The ADR SVM performs worse, giving the F1 score of 0.499.
### Conclusion
 Classical ML approaches can yield good performance when applied to ADR detection but require ADR-specific features engineered manually. Deep learning systems such as BERT, on the other hand, do not require feature engineering specific to the task, while still producing comparable results. A possible explanation is that pre-trained BERT model encapsulates “ADR-like” language knowledge and therefore can learn to detect ADRs by just being fine-tuned with ADR-specific training set. However, deep learning methods have a different concern – their learning complexity may be of a very high order and therefore requires significant computational resources, comparing to classical methods where the complexity is bounded by small number of features.

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

### BERT Classifier
#### Convert ADRMine data to BERT format
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

#### Fine-tune ADR model
Fine-tuning ADR model involves running bert neural network training and takes about 1 day on a fast Linux PC. That's why
Google compute engine with TPU (Tensorflow Processing Unit) is recommended where it takes around 1 hour. 
#### To run fine-tuning locally:
python adr_bert_classifier.py --vocab_file=bert_generic_model/uncased_L-24_H-1024_A-16/vocab.txt \
                       --bert_config_file=bert_generic_model/uncased_L-24_H-1024_A-16/bert_config.json \
                       --init_checkpoint=bert_generic_model/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                       --do_train=True --train_file=adrmine_data/adrmine_train.json --do_predict=True \
                       --predict_file=adrmine_data/adrmine_test.json --train_batch_size=24 \
                       --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=50 \
                       --output_dir=./bert_adr_model --use_tpu=False --version_2_with_negative=True

#### To run fine-tuning on TPU:
1. Create/Open Google VM Using Google TPU requires Google Cloud VM. It can be created from Google Cloud console (see https://blog.goodaudience.com/how-to-use-google-cloud-tpus-177c3a025067)
 NOTE: when creating VM, make sure to select TensorFlow ver. 1.11 or newer.
 
2. Create TPU Instance TPU instance is created from VM console (see https://blog.goodaudience.com/how-to-use-google-cloud-tpus-177c3a025067).
NOTE: Using TPU is not free, check on the pricing here: https://cloud.google.com/tpu/docs/pricing.

3. Create Google Storage Bucket This is required for feeding data in/out of TPU. (see https://blog.goodaudience.com/how-to-use-google-cloud-tpus-177c3a025067).

4. Start VM created in step 1, the fine-tuning process has to be done in it so that TPU can access resources.
All the next steps assume you are running inside the VM.

5. Configure ENV
```
export BERT_BASE_DIR=gs://squad-nn/bert/models/uncased_L-12_H-768_A-12
export SQUAD_DIR=/home/[your VM user name]/bert/squad_dir
export TPU_NAME=[TPU instance created in Step 2 above]
```

6. Run ADR Bert fine-tuning and prediction The following command will do the training and generate prediction files:
```
 python adr_bert_classifier.py --vocab_file=$BERT_LARGE_DIR/vocab.txt   --bert_config_file=$BERT_LARGE_DIR/bert_config.json   --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt   --do_train=True   --tra
in_file=$SQUAD_DIR/train-v2.0.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v2.0.json   --train_batch_size=24   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc
_stride=128   --output_dir=gs://squad-nn/bert/squad_large/   --use_tpu=True   --tpu_name=$TPU_NAME   --version_2_with_negative=True --do_lower_case=False
```
                       
#### Evaluating ADR model
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
python adr-evaluate.py adrmine_data/adrmine_test.json bert_adr_model/predictions.json \
                --na-prob-file bert_adr_model/null_odds.json
```

### SVM Classifier
```
python3.6 adr_svm_classifier.py --train-adrmine-tweets adrmine_data/download_tweets/train_tweet_posts.tsv \
                      --train-adrmine-annotations adrmine_data/download_tweets/train_tweet_annotations.tsv \
                      --test-adrmine-tweets adrmine_data/download_tweets/test_tweet_posts.tsv \
                      --test-adrmine-annotations adrmine_data/download_tweets/test_tweet_annotations.tsv \
                      --adrmine-adr-lexicon adrmine_data/ADR_lexicon.tsv
```