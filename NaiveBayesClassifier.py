import math
import re, nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from PreProcess import PreProcess
from FeatureSelection import FeatureSelection

class NaiveBayesClassifier:
    def __init__(self,train_df,classes) -> None:
        self.train_df = train_df
        self.classes = classes
        self.priors = self.get_prior_probs()
        return_val = self.get_class_counts()
        self.class_count_dicts = return_val[0]
        self.class_word_count = return_val[1]
        self.distinct_words = return_val[2]

    def get_prior_probs(self):
        priors = {}
        total_count = len(self.train_df)
        for i in range(self.classes):
            count = len( self.train_df[self.train_df['Sentiment'] == i] )
            priors.update({i:count/total_count})
        return priors
    
    def get_class_counts(self):
        class_count_dicts = [{} for i in range(self.classes)]
        class_word_count = [0]*self.classes
        distinct_words = set()

        for i in range(self.classes):
            mask = self.train_df['Sentiment'] == i
            for sentence in self.train_df[mask]['Phrase'].to_list():
                words = sentence.split(' ')
                for word in words:
                    count = class_count_dicts[i].get(word)
                    if not count:
                        class_count_dicts[i].update({word:1})
                    else:
                        count += 1
                        class_count_dicts[i].update({word:count})
                    class_word_count[i] += 1
                    distinct_words.add(word)
        return class_count_dicts, class_word_count, distinct_words

    def get_likelihood(self,word,sentiment_class):
        count = self.class_count_dicts[sentiment_class].get(word) 
        if not count:
            count = 0
        total_words_in_class = self.class_word_count[sentiment_class]
        likelihood = (count + 1)/ (total_words_in_class + len(self.distinct_words))
        return likelihood

    def classify_phrase(self,phrase):
        prosteriors = [1]*self.classes
        for i in range(self.classes):
            words = phrase.split(' ')
            for word in words:
                if word in self.distinct_words and word != '':
                    prosterior = self.get_likelihood(word,i) * self.priors.get(i)
                    prosteriors[i] *= prosterior
        return np.argmax(prosteriors)
    
    def classify(self,df):
        results = {'SentenceId':[],'Sentiment':[]}
        for i, sentence in enumerate(df['Phrase']):
            prediction = self.classify_phrase(sentence)
            sentence_id = df.iloc[i]['SentenceId']

            ids = results.get('SentenceId')
            ids.append(sentence_id)
            results.update({'SentenceId':ids})

            preds = results.get('Sentiment')
            preds.append(prediction)
            results.update({'Sentiment':preds})

        results = pd.DataFrame(results)
        return results

# pp = PreProcess('./moviereviews/train.tsv', './moviereviews/dev.tsv','./moviereviews/test.tsv',5)
# dfs = pp.return_processed_dfs()   

# nb = NaiveBayesClassifier(dfs[0],5)
# print(nb.index)
# print(nb.class_word_count)
# # res = nb.classify(dfs[1])


        
    
    
        