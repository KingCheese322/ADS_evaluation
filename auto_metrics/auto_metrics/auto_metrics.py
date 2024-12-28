import os
import numpy as np
import pandas as pd
from typing import Dict
from evaluate import load

class AutoEvaluator:
    def __init__(self):

        self.rouge = load("rouge")
        self.bleu = load("bleu")
        self.meteor = load("meteor")
        self.bertscore = load("bertscore")
        
    def read_file(self, file_path: str) -> str:

        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    
    def evaluate_pair(self, reference: str, prediction: str) -> Dict:

        results = {}
        
        rouge_scores = self.rouge.compute(
            predictions=[prediction], 
            references=[reference]
        )
        results.update({
            'rouge1_f1': rouge_scores['rouge1'],
            'rouge2_f1': rouge_scores['rouge2'],
            'rougeL_f1': rouge_scores['rougeL'],
            'rougeLsum_f1': rouge_scores['rougeLsum']
        })

        bleu_score = self.bleu.compute(
            predictions=[prediction], 
            references=[reference]
        )
        results['bleu'] = bleu_score['bleu']
        
        meteor_score = self.meteor.compute(
            predictions=[prediction], 
            references=[reference]
        )
        results['meteor'] = meteor_score['meteor']
        
        bert_score = self.bertscore.compute(
            predictions=[prediction], 
            references=[reference], 
            lang="en"
        )
        results.update({
            'bertscore_precision': np.mean(bert_score['precision']),
            'bertscore_recall': np.mean(bert_score['recall']),
            'bertscore_f1': np.mean(bert_score['f1'])
        })
        
        return results
    
    
    def read_file(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            
            return f.read().decode('utf-8', errors='replace')