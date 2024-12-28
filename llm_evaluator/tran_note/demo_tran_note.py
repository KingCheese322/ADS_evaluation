import os
os.chdir("/Users/rui/Desktop/llm_evaluator/tran_note")

from evaluator_tran_note import MedicalNoteEvaluator
    
transcript_path = "../llm_evaluator/data/whisper_transcript/DC0001 - DPt0411 edited_deidentified.txt"
note_path = "..//llm_evaluator/data/gpt_note/DC0001 - DPt0411 edited_deidentified_note.txt"
    
evaluator = MedicalNoteEvaluator(
    api_key="API_KEY",
    model_name="gpt-4o",
    temperature=0.7
    )
    
result = evaluator.evaluate_note(transcript_path, note_path)