import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Evaluate:
    def __init__(self,predictions,ground_truths,classes) -> None:
        self.pred = predictions
        self.gt = ground_truths
        self.classes = classes
    
    def get_f1(self):
        all_f1 = []
        for i in range(self.classes):
            tp = len(self.pred[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] == i)])
            fn = len(self.pred[(self.pred['Sentiment'] != i) & (self.gt['Sentiment'] == i)])
            fp = len(self.pred[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] != i)])
            f1 = (2*tp)/(2*tp + (fp+fn) )
            all_f1.append(f1)
        return sum(all_f1) / self.classes

    def get_confusion(self):
        matrix = np.zeros((self.classes,self.classes))
        for i in range(self.classes):
            for j in range(self.classes):
                matrix[i][j] = len(self.gt[(self.pred['Sentiment'] == i) & (self.gt['Sentiment'] == j)])
        cm_vis = sns.heatmap(matrix, annot=True, cmap='Blues', fmt='.3g')
        plt.show()

        return cm_vis
    
    def get_accuracy(self):
        tp = len(self.pred[self.pred['Sentiment'] == self.gt['Sentiment']])
        #print(len(self.pred[(self.pred['Sentiment'] == 1)]))
        return tp/len(self.pred)

