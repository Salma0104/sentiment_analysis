# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
from PreProcess import PreProcess
from FeatureSelection import FeatureSelection
from NaiveBayesClassifier import NaiveBayesClassifier
from Evaluate import Evaluate
from datetime import date, datetime

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acb19sh" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    tn = datetime.now()
    pp = PreProcess(training, dev,test,number_classes)
    dfs = pp.return_processed_dfs()

    if features == 'all_words':
        f_train = dfs[0]
        f_dev = dfs[1]
    else:
        fs = FeatureSelection(['adjective','noun','verb','adverb'])
        # # f_train = fs.filter_by_features(dfs[0])
        # f_train = fs.filter_by_polarity(dfs[0])
        # #f_train = fs.filter_by_subjectivity(dfs[0])
        f_dev = dfs[1]
        f_train = dfs[0]
        f_words = fs.get_most_subjective(dfs[0])

    nb_classifier = NaiveBayesClassifier(f_train,number_classes,True,f_words)
    results = nb_classifier.classify(f_dev)

    if output_files:
        # either 'dev' or 'test' If test change f_dev to f_test in nb_classifier.classify()
        mode = 'dev' 
        name = f'{mode}_predictions_{number_classes}classes_{USER_ID}.tsv'
        results.to_csv(name, sep="\t",mode='w',index=False)

    ev = Evaluate(results,f_dev,number_classes)
    
    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = ev.get_f1()

    if confusion_matrix:
        ev.get_confusion()
    
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))
    print(f'Program took: {datetime.now()- tn} seconds to run')

if __name__ == "__main__":
    main()