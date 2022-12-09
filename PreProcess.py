from typing import List
import pandas as pd
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class PreProcess:
    def __init__(self,train_path,dev_path,test_path,classes) -> None:
        self.train_df = self.load_data_as_df(train_path)
        self.dev_df = self.load_data_as_df(dev_path)
        self.test_df = self.load_data_as_df(test_path)
        self.classes = classes

    def load_data_as_df(self,file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t')
    
    def process(self,df: pd.DataFrame, column: str) -> pd.DataFrame:
        # excludes exclamation mark
        punc = string.punctuation[1:]
        # remove all punctiation
        df[column] = df[column].astype(str).str.translate(str.maketrans('', '', punc))
        #change double space to single space
        df[column] = df[column].astype(str).str.replace('  ',' ')
        # make text lowercase
        df[column] = df[column].astype(str).str.lower()

        return df
    
    def map_5_to_3(self,df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df['Sentiment'] == 1, 'Sentiment'] = 0
        df.loc[df['Sentiment'] == 2, 'Sentiment'] = 1
        df.loc[df['Sentiment'] == 3, 'Sentiment'] = 2
        df.loc[df['Sentiment'] == 4, 'Sentiment'] = 2
        return df
    
    def apply_stemming(self, df: pd.DataFrame):
        stemmer = SnowballStemmer("english")
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split(' ')
            filtered_words = []
            for j, word in enumerate(words):
                filtered_words.append(stemmer.stem(word))
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df
    
    def apply_stopwords(self,df: pd.DataFrame):
        stopword_list = set(stopwords.words('english'))
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split(' ')
            filtered_words = []
            for j, word in enumerate(words):
                if word not in stopword_list:
                    filtered_words.append(words[j].strip())
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df

    def return_processed_dfs(self) -> List[pd.DataFrame]:
        dfs = [self.train_df,self.dev_df,self.test_df]
        for i, df in enumerate(dfs):
            dfs[i] = self.process(df,'Phrase')
            dfs[i] = self.apply_stopwords(dfs[i])
            if self.classes == 3 and i != 2:
                dfs[i] = self.map_5_to_3(dfs[i])
        return dfs

        


