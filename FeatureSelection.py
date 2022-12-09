import pandas as pd
import re, nltk
from PreProcess import PreProcess
from textblob import TextBlob


class FeatureSelection:
    def __init__(self,features) -> None:
        self.features = features
        self.tag_dict = {'adjective':['JJ','JJR','JJS'],'noun':['NN','NNS','NNP','NNPS'],'adverb':['RB','RBS','RBR','RP'],'verb':['VB','VBD','VBG','VBN','VBP','VBZ']}
        self.tag_set = self.feature_to_set()
    
    def feature_to_set(self):
        tag_set = []
        for feature in self.features:
            tag_set += self.tag_dict.get(feature)
        tag_set = set(tag_set)
        return tag_set
    
    def get_tagged_phrases(self,df):
        tag = lambda x: nltk.pos_tag(x.split(' '))
        tagged_phrases = list(map(tag,df['Phrase'].to_list()))
        return tagged_phrases
    
    def filter_by_features(self, df):
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

    def filter_by_polarity(self,df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split(' ')
            filtered_words = []
            for j, word in enumerate(words):
                if TextBlob(word).sentiment.polarity != 0: 
                    filtered_words.append(words[j])
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df

    def filter_by_subjectivity(self,df: pd.DataFrame):
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split(' ')
            filtered_words = []
            for j, word in enumerate(words):
                if abs(TextBlob(word).sentiment.subjectivity) > 0.3: 
                    filtered_words.append(words[j])
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df
        
        