from typing import List
import pandas as pd
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class PreProcess:
    ''' Applies Pre-processing steps to all data
        
        Args:
            train_path (str): A string containing the path of the train.tsv file
            dev_path (str): A string containing the path of the dev.tsv file
            test_path (str): A string containing the path of the test.tsv file
            classes (int): An integer denoting the number of classes. 5 or 3
        
        Attributes:
            train_df (pd.DataFrame): A Pandas DataFrame containing train data
            dev_df (pd.DataFrame): A Pandas DataFrame containing dev data
            test_df (pd.DataFrame): A Pandas DataFrame containing dev data
            classes (int): An integer denoting the number of classes
    '''
    def __init__(self,train_path,dev_path,test_path,classes) -> None:
        self.train_df = self.load_data_as_df(train_path)
        self.dev_df = self.load_data_as_df(dev_path)
        self.test_df = self.load_data_as_df(test_path)
        self.classes = classes
    
    ''' Returns a tsv thats been loaded into a pandas DataFrame'''
    def load_data_as_df(self,file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t')
    
    ''' Reurns a pandas DataFrame that has had its Phrase column preprocessed'''
    def process(self,df: pd.DataFrame, column: str) -> pd.DataFrame:
        #refers to all punctuation except !
        punc = string.punctuation[1:]
        # remove all punctiation
        df[column] = df[column].astype(str).str.translate(str.maketrans('', '',punc))
        # make text lowercase
        df[column] = df[column].astype(str).str.lower()
        return df
    
    '''
        Returns a pandas DataFrame that has had its sentiment column
        Alterered to there are 3 classes in total    
    '''
    def map_5_to_3(self,df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df['Sentiment'] == 1, 'Sentiment'] = 0
        df.loc[df['Sentiment'] == 2, 'Sentiment'] = 1
        df.loc[df['Sentiment'] == 3, 'Sentiment'] = 2
        df.loc[df['Sentiment'] == 4, 'Sentiment'] = 2
        return df
    
    ''' Returns pandas DataFrame with Phrase column being stemmed'''
    def apply_stemming(self, df: pd.DataFrame):
        stemmer = SnowballStemmer("english")
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            filtered_words = []
            for j, word in enumerate(words):
                filtered_words.append(stemmer.stem(word))
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df

    ''' Returns pandas DataFrame where stopwords are removed from Phrase column'''
    def apply_stopwords(self,df: pd.DataFrame):
        stopword_list = set(stopwords.words('english'))
        phrases = df['Phrase'].to_list()
        for i, sentence in enumerate(phrases):
            words = sentence.split()
            filtered_words = []
            for j, word in enumerate(words):
                if word not in stopword_list:
                    filtered_words.append(words[j].strip())
            phrases[i] = ' '.join(filtered_words)
        df['Phrase'] = phrases
        return df
    
    ''' Returns a list of pandas Dataframes where preeprocessing and 
        Stopword removal have been applied
    '''
    def return_processed_dfs(self) -> List[pd.DataFrame]:
        dfs = [self.train_df,self.dev_df,self.test_df]
        for i, df in enumerate(dfs):
            dfs[i] = self.process(df,'Phrase')
            dfs[i] = self.apply_stopwords(dfs[i])
            #Not being used as its not useful
            #dfs[i] = self.apply_stemming(dfs[i])
            if self.classes == 3 and i != 2:
                # Map ground truth except for test data
                dfs[i] = self.map_5_to_3(dfs[i])
        return dfs

        


