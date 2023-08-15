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
from datetime import datetime

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
    
    ''' Returns the prediction and ground truth
        Complete end to end workflow, preprocessing -> classification
        '''
    def complete_workflow(training,dev,test,classes,feature,mode):
        # mode is either 1 or 2 -> correspond to either dev or test
        pp = PreProcess(training,dev,test,classes)
        dfs = pp.return_processed_dfs()
        if feature != 'all_words':
            fs = FeatureSelection(['adjective','noun','verb','adverb'])
            f_train = fs.filter_by_word_type(dfs[0])
        else:
            f_train = dfs[0]
        nb_classifier = NaiveBayesClassifier(f_train,classes)
        results = nb_classifier.classify(dfs[mode])
        return results,dfs[mode]
    
    # classify for all modes and classes and write them to a file
    if output_files:
        mode = [1,2]
        mode_name = {1:'dev',2:'test'}
        classes =[3,5]  
        for i in range(len(mode)):
            for j in range(len(classes)):
                result = complete_workflow(training,dev,test,classes[j],'features',mode[i])[0]
                name = f'{mode_name.get(mode[i])}_predictions_{classes[j]}classes_{USER_ID}.tsv'
                result.to_csv(name, sep="\t",mode='w',index=False)
    # Classify dev set for specified config
    else:
        output = complete_workflow(training,dev,test,number_classes,features,1)
        results = output[0]
        ev = Evaluate(results,output[1],number_classes)
        f1_score = ev.get_f1()
        accuracy = ev.get_accuracy()

        if confusion_matrix:
            ev.get_confusion()

        """
        IMPORTANT: your code should return the lines below. 
        However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
        """
        if features == 'all_words':
            features = False
        else:
            features = True
        print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
        print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()