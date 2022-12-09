import pandas as pd
from PreProcess import PreProcess
import numpy as np

class Test:

    def __init__(self,df) -> None:
        self.df = df
    
    def counts_dict(self):
        count_dict = {}
        for phrase in self.df['Phrase'].to_list():
            for word in phrase.split(' '):
                count = count_dict.get(word)
                if not count:
                    count_dict.update({word:1})
                else:
                    count += 1
                    count_dict.update({word:count})
        counts = [x[1] for x in count_dict.items()]
        return np.average(counts)

pp = PreProcess('./moviereviews/train.tsv', './moviereviews/dev.tsv','./moviereviews/test.tsv',5)
dfs = pp.return_processed_dfs() 

t = Test(dfs[0])
print(t.counts_dict())