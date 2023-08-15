import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Evaluate:

    ''' Evaluates the performance of a prediction
        
        Args:
            pred (pd.DataFrame): A pandas dataframe containing the sentiment 
                prediction for each phrase/sentence 
            gt (pd.DataFrame): A pandas dataframe containing the sentiment 
                ground truth for each phrase/sentence 
            classes (int): An integer denoting the number of classes. 5 or 3
        
        Attributes:
            pred (pd.DataFrame): A pandas dataframe containing the sentiment 
                prediction for each phrase/sentence 
            gt (pd.DataFrame): A pandas dataframe containing the sentiment 
                ground truth for each phrase/sentence 
            classes (int): An integer denoting the number of classes. 5 or 3
    '''
    def __init__(self,predictions,ground_truths,classes) -> None:
        self.pred = predictions
        self.gt = ground_truths
        self.classes = classes
    
    ''' Returns the F1 score for the pred DataFrame'''
    def get_f1(self):
        all_f1 = []
        for i in range(self.classes):
            tp = len(self.pred[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] == i)])
            fn = len(self.pred[(self.pred['Sentiment'] != i) & (self.gt['Sentiment'] == i)])
            fp = len(self.pred[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] != i)])
            f1 = (2*tp)/(2*tp + (fp+fn) )
            all_f1.append(f1)
        return sum(all_f1) / self.classes

    ''' Plots the confusion matrix for the pred and gt'''
    def get_confusion(self):
        matrix = np.zeros((self.classes,self.classes))
        for i in range(self.classes):
            for j in range(self.classes):
                matrix[i][j] = len(self.gt[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] == j)])
        cm_vis = sns.heatmap(matrix, annot=True, cmap='Blues', fmt='.3g')
        plt.show()

    ''' Returns the accuracy of predictions'''
    def get_accuracy(self):
        tp = len(self.pred[self.pred['Sentiment'] == self.gt['Sentiment']])
        return tp/len(self.pred)

