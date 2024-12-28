from evaluate import load

class PerplexityEvaluator:
    def __init__(self):

        self.perplexity = load("perplexity", module_type="metric")
        
    def read_file(self, file_path: str) -> str:

        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    
    def evaluate_single(self, text: str) -> float:

        return self.perplexity.compute(predictions=text, model_id="gpt2")["mean_perplexity"]