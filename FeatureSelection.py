import pandas as pd
import nltk
from textblob import TextBlob
import numpy as np


class FeatureSelection:
    ''' Applies Feature extraction to a pandas DataFrame
        
        Args:
            features (list): a list of features (e.g. noun/verb..) that will be used for filtering
        
        Attributes:
            features (list): a list of features (e.g. noun/verb..) that will be used for filtering
            tag_dict (dict): a dictionary mapping a word type to its corresponding pos-tag values
            tag_set (set): a set containing all the relevent pos-tags
    '''
    def __init__(self,features) -> None:
        self.features = features
        self.tag_dict = {'adjective':['JJ','JJR','JJS'],'noun':['NN','NNS','NNP','NNPS'],'adverb':['RB','RBS','RBR','RP'],'verb':['VB','VBD','VBG','VBN','VBP','VBZ']}
        self.tag_set = self.feature_to_set()
    
    '''Returns a set containing the required pos-tags'''
    def feature_to_set(self):
        tag_set = []
        for feature in self.features:
            tag_set += self.tag_dict.get(feature)
        tag_set = set(tag_set)
        return tag_set
    
    ''' Returns a list containing the Phrases column of a 
        pandas DataFrame where each word in a phrase is 
        converted to (word,pos-tag) e.g 'salma' -> ('salma','NN')
    '''
    def get_tagged_phrases(self,df):
        tag = lambda x: nltk.pos_tag(x.split())
        tagged_phrases = list(map(tag,df['Phrase'].to_list()))
        return tagged_phrases
    
    ''' Returns pandas DataFrame where words thats post-tag is not
        present in the tag-set are removed
    '''
    def filter_by_word_type(self, df):
        if len(self.tag_set) > 0: 
            phrases = self.get_tagged_phrases(df)
            for i, phrase in enumerate(phrases):
                filtered_words = []
                for j, (word,tag) in enumerate(phrase):
                    if tag in self.tag_set:
                        filtered_words.append(word)
                phrases[i] = ' '.join(filtered_words)
            df['Phrase'] = phrases
        return df
    
    ''' Returns pandas DataFrame where words thats dont 
        have polarity are removed
    '''
    def filter_by_polarity(self,df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            filtered_words = []
            for j, word in enumerate(words):
                if abs(TextBlob(word).sentiment.polarity) != 0: 
                    filtered_words.append(words[j])
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df

    ''' Returns pandas DataFrame where words thats dont 
        have subjectivity are removed
    '''
    def filter_by_subjectivity(self,df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            filtered_words = []
            for j, word in enumerate(words):
                if TextBlob(word).sentiment.subjectivity > 0: 
                    filtered_words.append(words[j])
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df
    
    ''' Returns a list of words ranked by there subjectivity in descending order'''
    def get_most_subjective(self, df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        # in the form of [(word,subj_scrore)]
        subjective = []
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            for j, word in enumerate(words):
                subjective.append((word,TextBlob(word).sentiment.subjectivity))
        subjective = list(sorted(subjective,key=lambda x: x[1], reverse=True))
        subjective_words = [x[0] for x in subjective]
        return subjective_words
    
    ''' Returns pandas DataFrame where words not in the top
        74% most subjective list are removed 
    '''
    def filter_by_most_subjective(self,df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        top_subj = self.get_most_subjective(df)
        top_subj = set(top_subj[:int(len(top_subj)*0.74)])
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            filtered_words = []
            for j, word in enumerate(words):
                if word in top_subj: 
                    filtered_words.append(words[j])
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df
        
        