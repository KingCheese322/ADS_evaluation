import re
import pandas as pd
from langchain_openai import ChatOpenAI

class MedicalFactExtractor:
    def __init__(self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.7):

        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        self.FACT_PROMPT = """
        ###TASK
        Extract atomic facts from the medical content. 
        Each fact should be the smallest, indivisible piece of clinical information.

        ###GUIDELINES
        1. Each fact should contain exactly ONE piece of clinical information.
        2. Facts must be self-contained and context-independent.
        3. Remove redundant information.
        4. Preserve temporal and contextual qualifiers when clinically relevant.
        5. Maintain numerical values and units exactly as stated.

        ###EXAMPLES
        GOOD (Atomic):
        Patient experiences headaches three times per week.
        Blood pressure reading was 140/90 mmHg.
        Patient takes 10mg Lisinopril daily.

        BAD (Not Atomic):
        Patient has headaches three times per week and feels nauseous (Should be split into two facts).
        Patient's vitals were normal (Too vague, should specify each vital sign).
        Patient takes medications for blood pressure (Should specify medication and dosage).

        ### Medical Content
        Please extract facts from the content and output them in JSON format:
        {input}

        ###OUTPUT FORMAT
        Return facts in the following format:
        {{
        "facts": [
        }}
        "fact_id": number,
        "content": string,
        }}
        ]
        }}
        """
    
    def read_file(self, file_path: str) -> str:

        with open(file_path, 'rb') as f:

            return f.read().decode('utf-8', errors='replace')
    
    def extract_facts_from_text(self, text: str) -> pd.DataFrame:

        response = self.llm.invoke(self.FACT_PROMPT.format(input=text)).content
        
        pattern = r'"fact_id":\s*(\d+),\s*"content":\s*"([^"]+)"'
        matches = re.finditer(pattern, response)
        
        facts = []
        for match in matches:
            facts.append({
                'fact_id': int(match.group(1)),
                'content': match.group(2)
            })
        
        return pd.DataFrame(facts)
    
    def extract_facts_from_file(self, file_path: str) -> pd.DataFrame:

        text = self.read_file(file_path)
        
        return self.extract_facts_from_text(text)