import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    ''' Applies Feature extraction to a pandas DataFrame
        
        Args:
            train_df (pd.DataFrame): A Pandas DataFrame containing train data
            classes (int): An integer denoting the number of classes. 5 or 3
        
        Attributes:
            priors (dict): A dictionary containing the prior value for each class
            class_count_dicts (list): A list that contains a dictionary where each
                dictionary has the count of each word in that class
            class_word_count (list): A list containing total word count for each class
            distinct_words (set): A set that contains all the distinc words present in
                all classes
    '''
    def __init__(self,train_df,classes) -> None:
        self.train_df = train_df
        self.classes = classes
        # in the form {class:prior_val}
        self.priors = self.get_prior_probs()
        self.class_count_dicts, self.class_word_count, self.distinct_words = \
            self.get_class_counts()
    
    ''' Returns a dictionary mapping class to prior probability'''
    def get_prior_probs(self):
        priors = {}
        total_count = len(self.train_df)
        for i in range(self.classes):
            # Number of phrases thats senntiment value is i where i is a class
            count = len( self.train_df[self.train_df['Sentiment'] == i] )
            priors.update({i:count/total_count})
        return priors
    
    ''' Returns: 
            A list containing dictionaries Mapping word to count for a class
            A list containing the count of words for each class, where list[i]
                contains count of words for class i
            A set of all the distinct words present 
    '''
    def get_class_counts(self):
        # in the form [{word:count}]
        class_count_dicts = [{} for i in range(self.classes)]
        class_word_count = [0]*self.classes
        distinct_words = set()

        for i in range(self.classes):
            mask = self.train_df['Sentiment'] == i
            for sentence in self.train_df[mask]['Phrase'].to_list():
                words = sentence.split()
                for word in words:
                    count = class_count_dicts[i].get(word)
                    #if word doesnt exist in dict add it to the dict
                    if not count:
                        class_count_dicts[i].update({word:1})
                    # if it does exist, update the count and then update the dict
                    else:
                        count += 1
                        class_count_dicts[i].update({word:count})
                    # increment by 1 for each word present
                    class_word_count[i] += 1
                    # for each word add it to the set
                    distinct_words.add(word)
        return class_count_dicts, class_word_count, distinct_words

    ''' Returns the likelihood for a given word in a class'''
    def get_likelihood(self,word,sentiment_class):
        count = self.class_count_dicts[sentiment_class].get(word) 
        if not count:
            count = 0
        total_words_in_class = self.class_word_count[sentiment_class]
        # Uses lapace smoothing
        likelihood = (count + 1)/ (total_words_in_class + len(self.distinct_words))
        return likelihood
    
    ''' Returns the class that the phrase most likely belongs to'''
    def classify_phrase(self,phrase):
        prosteriors = [1]*self.classes
        for i in range(self.classes):
            words = phrase.split()
            for word in words:
                if word in self.distinct_words and word != '':
                    # Since class is unbalanced, ignore prior prob
                    prosterior = self.get_likelihood(word,i) #* self.priors.get(i)
                    prosteriors[i] *= prosterior
        return np.argmax(prosteriors)
    
    ''' Returns a pandas DataFrame containing a column of sentednceIds 
        and the corresponding prediction in anoter column'''
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




        
    
    
        