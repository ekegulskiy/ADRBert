# ADRBert
## Overview
### Objective
 Social Media has undoubtfully become one of the major sources of information over the last decade. Much research has been done to study how users’ posts can help monitor Adverse Drug Reaction (ADR) cases and what can be done to prevent them. One of such research projects developed a system called ADRMine [3], producing a state-of-the-art ADR detection system with F1-scores of 0.821 and 0.721 for DailyStrength and Twitter data respectively [3, Table 3]. The system is based on a machine learning method called Conditional Random Fields (CRF), belonging to the “sequence modeling family” of “probabilistic graphical models” [7]. The objective of my project is to apply a deep learning technique to ADR detection and compare to classical machine learning approaches such as CRF and SVM.
### Methods
 There are 2 methods I use in the project to compare deep learning ADR detection approach to the classical ones. First, a deep learning classifier is developed, ADRBert, using BERT system [1] from recent Google research. BERT is a neural network system based on learning and encoding general language representation model from very large text corpus. First, I fine-tune BERT’s model with ADRMine training data and evaluate the performance with ADRMine test data. Second, I develop simple SVM Classifier, ADR SVM, to serve as a baseline representing classical machine learning approaches. The ADR SVM is using 3 manually engineered features – a sub-set of features used in ADRMine system.
### Results
 ADRBert yields a decent F1 score of 0.702 for Twitter data which is just slightly lower than ADRMine F1 score of 0.721 for the same data. The ADR SVM performs worse, giving the F1 score of 0.499.
### Conclusion
 Classical ML approaches can yield good performance when applied to ADR detection but require ADR-specific features engineered manually. Deep learning systems such as BERT, on the other hand, do not require feature engineering specific to the task, while still producing comparable results. A possible explanation is that pre-trained BERT model encapsulates “ADR-like” language knowledge and therefore can learn to detect ADRs by just being fine-tuned with ADR-specific training set. However, deep learning methods have a different concern – their learning complexity may be of a very high order and therefore requires significant computational resources, comparing to classical methods where the complexity is bounded by small number of features.
### References
[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805v1 [cs.CL] 11 Oct 2018

[2] Ioannis Korkontzelos, Azadeh Nikfarjam, Matthew Shardlow, et al., Analysis of the effect of sentiment analysis on extracting adverse drug reactions from tweets and forum posts, Journal of Biomedical Informatics Volume 62, August 2016, Pages 148-158

[3] A. Nikfarjam, A. Sarker, K. O’Connor, R. Ginn, G. Gonzalez, Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling with word embedding cluster features, J. Am. Med Inform. Assoc. (2015), http://dx.doi.org/10.1093/jamia/ocu041.

[4] FDA Adverse Event Reporting System (FAERS) Public Dashboard. https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard. Accessed May 2019.

[5] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang. (2016) SQuAD: 100,000+ Questions for Machine Comprehension of Text.  arXiv:1606.05250 [cs.CL]

[6] Source code used for this research: https://github.com/ekegulskiy/ADRBert 

[7] Wikipedia, Conditional Random Fields definition https://en.wikipedia.org/wiki/Conditional_random_field

[8] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez et al., Attention Is All You Need, arXiv:1706.03762v5 [cs.CL] 6 Dec 2017

[9] Python ML Tools and Classifiers, https://scikit-learn.org/stable/

[10] Barnabás Póczos, Introduction to Machine Learning CMU-10701, https://alex.smola.org/teaching/cmu2013-10-701/slides/11_Learning_Theory.pdf 

[11] Devroye et al., 1996; Vapnik, 1998, Vapnik–Chervonenkis Theory, Error Estimation for Pattern Recognition, First Edition. Ulisses M. Braga-Neto and Edward R. Dougherty.© 2015 The Institute of Electrical and Electronics Engineers, Inc. Published 2015 by John Wiley & Sons, Inc (https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119079507.app2) 

[12] BERT GitHub repository, https://github.com/google-research/bert


## Instructions
### Python Requirements
1. Both python3.6 and python2.7 version required, due to ADRMine scripts using python2.7
2. sudo pip3.6 install gitpython
3. sudo apt-get install python3-git
4. sudo pip3.6 install -r requirements.txt

### Setup Project Environment
To setup project environments, execute setup_env.py script as following:
```
python3.6 setup_env.py
```

It will download and prepare the following sub-directories with components:
1. adrmine_data (ADRMine data from http://diego.asu.edu/downloads/publications/ADRMine/download_tweets.zip)
2. bert (GitHub repo from https://github.com/google-research/bert.git)
3. bert_generic_model (bert pre-trained model from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)
4. bert_adr_out (bert fine-tuned ADR model)

### ADR BERT Classifier
#### Convert ADRMine data to BERT SQuAD format
1. Use ADRMine script to download training and test set tweets from which ADR annotation were created:

   **NOTE 1: ADRMine script requires python2.7**

   **NOTE 2: This will take ~5 minutes as it downloads tweets from Twitter.com**

```
python2.7 adrmine_data/download_tweets/download_tweets.py \
       adrmine_data/download_tweets/test_tweet_ids.tsv > adrmine_data/download_tweets/test_tweet_posts.tsv

python2.7 adrmine_data/download_tweets/download_tweets.py \
       adrmine_data/download_tweets/train_tweet_ids.tsv > adrmine_data/download_tweets/train_tweet_posts.tsv
```

2. Convert ADRMine training and test sets into BERT SQuAD JSON format:

```
python3.6 generate_bert_data.py --adrmine-tweets=adrmine_data/download_tweets/test_tweet_posts.tsv \
                       --adrmine-annotations=adrmine_data/download_tweets/test_tweet_annotations.tsv \
                       --json-output-file=adrmine_data/adrmine_test.json

python3.6 generate_bert_data.py --adrmine-tweets=adrmine_data/download_tweets/train_tweet_posts.tsv \
                       --adrmine-annotations=adrmine_data/download_tweets/train_tweet_annotations.tsv \
                       --json-output-file=adrmine_data/adrmine_train.json

```

#### Fine-tune ADR BERT model
Fine-tune ADR BERT model by re-training BERT pre-trained model with ADRMine data.

**NOTE 1: Fine-tuning ADR model involves running bert neural network training and takes about 1 day on a fast Linux PC. That's why
Google compute engine with TPU (Tensorflow Processing Unit) is recommended where it takes around 1 hour.**

**NOTE 2: Fine-tuning has been already done using TPU and the model is saved in "bert_adr_model" folder. So running this step is not required by any other steps.**

#### To run fine-tuning locally:
```
python3.6 adr_bert_classifier.py --vocab_file=bert_generic_model/uncased_L-24_H-1024_A-16/vocab.txt \
                       --bert_config_file=bert_generic_model/uncased_L-24_H-1024_A-16/bert_config.json \
                       --init_checkpoint=bert_generic_model/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                       --do_train=True --train_file=adrmine_data/adrmine_train.json --do_predict=True \
                       --predict_file=adrmine_data/adrmine_test.json --train_batch_size=24 \
                       --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=50 \
                       --output_dir=./bert_adr_model --use_tpu=False --version_2_with_negative=True
```

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

6. Run ADR BERT fine-tuning and prediction. The following command will do the training and generate prediction files:
```
 python3.6 adr_bert_classifier.py --vocab_file=$BERT_LARGE_DIR/vocab.txt   --bert_config_file=$BERT_LARGE_DIR/bert_config.json   --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt   --do_train=True   --tra
in_file=$SQUAD_DIR/train-v2.0.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v2.0.json   --train_batch_size=24   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc
_stride=128   --output_dir=gs://squad-nn/bert/squad_large/   --use_tpu=True   --tpu_name=$TPU_NAME   --version_2_with_negative=True --do_lower_case=False
```
                       
#### Evaluating ADR BERT model
Evaluation of ADR BERT model can be run as on desktop Linux PC as it does not very long time (about 2-4 minutes).
However, it may run out of memory. On 16-GB Linux PC, it exceeded memory usage by 10% but still was able to run.

1. Evaluate test set and create predictions files (./bert_adr_model/predictions.json and null_odds.json) required for F1 computation:
```
python3.6 adr_bert_classifier.py --vocab_file=bert_generic_model/uncased_L-24_H-1024_A-16/vocab.txt \
                       --bert_config_file=bert_generic_model/uncased_L-24_H-1024_A-16/bert_config.json \
                       --init_checkpoint=bert_generic_model/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                       --do_train=False --train_file=adrmine_data/adrmine_train.json --do_predict=True \
                       --predict_file=adrmine_data/adrmine_test.json --train_batch_size=24 \
                       --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=50 \
                       --output_dir=./bert_adr_model --use_tpu=False --version_2_with_negative=True
```

2. Compute F1 score:
```
python3.6 adr_bert_evaluate.py adrmine_data/adrmine_test.json bert_adr_model/predictions.json \
                --na-prob-file bert_adr_model/null_odds.json
```

### ADR SVM Classifier
Run ADR SVM Classifier to train and evaluate ADRMine data:
```
python3.6 adr_svm_classifier.py --train-adrmine-tweets adrmine_data/download_tweets/train_tweet_posts.tsv \
                      --train-adrmine-annotations adrmine_data/download_tweets/train_tweet_annotations.tsv \
                      --test-adrmine-tweets adrmine_data/download_tweets/test_tweet_posts.tsv \
                      --test-adrmine-annotations adrmine_data/download_tweets/test_tweet_annotations.tsv \
                      --adrmine-adr-lexicon adrmine_data/ADR_lexicon.tsv
```