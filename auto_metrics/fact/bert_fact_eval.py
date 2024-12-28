import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIFactEvaluator:

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        self.model, self.tokenizer, self.device = self.setup_nli_model()

    def setup_nli_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device

    def get_nli_prediction(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        return probs[0].cpu().numpy()

    
    def evaluate_single_pair(self, pred_row, ref_row):
        pred_facts = json.loads(pred_row['facts_content'])
        ref_facts = json.loads(ref_row['facts_content'])
        
        supported_count = 0
        pred_facts_count = len(pred_facts)
        ref_facts_count = len(ref_facts)
        
        print(f"Processing {pred_facts_count} facts...")
        
        for pred_fact in tqdm(pred_facts):
            max_entailment = 0
            
            for ref_fact in ref_facts:
                probs = self.get_nli_prediction(ref_fact['content'], pred_fact['content'])
                entailment_prob = float(probs[0])
                if entailment_prob > max_entailment:
                    max_entailment = entailment_prob
            
            if max_entailment > self.threshold:
                supported_count += 1
        
        link_score = min(1.0, supported_count / ref_facts_count)
        correct_score = supported_count / pred_facts_count
        
        return {
            'file_name': pred_row['file_name'],
            'link_score': link_score,
            'correct_score': correct_score
        }