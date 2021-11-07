import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

def accuracy_table(target, prediction):

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        target, prediction, average='weighted')
    accuracy = accuracy_score(prediction, target)
    mcc = matthews_corrcoef(prediction, target)

    measures = ['Accuracy','F1_Score','Precision', 'Recall', 'MCC']
    scores = [accuracy, f1_score, precision, recall, mcc]
    
    accuracytable = pd.DataFrame({'Measure': measures, 'Value': scores})
    
    return accuracytable