# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:37:53 2024

@author: ch513
"""

from openai import AzureOpenAI
import os
import nltk
from nltk.tokenize import sent_tokenize
import re

 
openai_key = "3534d5388ac6429588e7d1ef4420ddb5"
openai_chat_endpoint = "https://dhd-aihealth-oai01.openai.azure.com/"
openai_chat_model = "aihealth-openai-model4o"
openai_api_version = '2024-02-15-preview'

nltk.download('punkt')


print(f"OpenAI Chat Endpoint: {openai_chat_endpoint}")
 
if not openai_key or not openai_chat_endpoint or not openai_chat_model:
 print('Please set the OPENAI_API_KEY, OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_MODEL environment variables.')
 quit()
 

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=openai_key,
    api_version=openai_api_version,
    azure_endpoint=openai_chat_endpoint
)

def truncate_transcript(text, max_words=1000):
    sentences = sent_tokenize(text)
    current_count = 0
    batch = []
    batches = []
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        # If adding this sentence exceeds the limit, store the current batch and start a new one
        if current_count + sentence_length > max_words:
            if batch:
                batches.append(' '.join(batch))
                batch = []
                current_count = 0
        batch.extend(sentence_words)
        current_count += sentence_length
    # Add the last batch if there's any remaining
    if batch:
        batches.append(' '.join(batch))
    return batches

def save_batches(batches, filename, batch_path):
    base, ext = os.path.splitext(filename)
    for i, batch in enumerate(batches):
        output_filename = f"{base}_batch{i+1}{ext}"
        output_path = os.path.join(batch_path, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(batch)
        print(f"Saved Batch {i+1} to {output_path}")
        
def identify_names(transcript_batch, client, openai_chat_model):
    instructions = """
    Below is a transcript of a conversation between a doctor and a patient. Identify the unique names in the conversation. Return the names as a comma-separated list.
    """
    user_content = instructions + '\n\n' + transcript_batch
    chat_completion = client.chat.completions.create(
        model=openai_chat_model,  # specify the correct model here
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with identifying names in the transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    # Extract the response content, assuming it returns names as a comma-separated list
    response = chat_completion.choices[0].message.content
    names = [name.strip() for name in response.split(',')]
    return set(names)  # Return unique names as a set


def replace_names_with_identifiers(batches, client, openai_chat_model):
    unique_names = set()
    
    # Step 1: Collect all unique names from all batches using OpenAI API
    for batch in batches:
        unique_names.update(identify_names(batch, client, openai_chat_model))

    # Step 2: Assign unique identifiers to each name
    name_mapping = {name: f"NAME{i+1}" for i, name in enumerate(unique_names)}

    # Step 3: Replace names in each batch with their respective identifier while preserving formatting
    updated_batches = []
    for batch in batches:
        # Preserve the line breaks and separators within each batch by splitting and re-joining
        lines = batch.splitlines()  # Split into lines to preserve structure
        updated_lines = []
        
        for line in lines:
            # Replace names in each line while preserving the line structure
            updated_line = line
            for name, identifier in name_mapping.items():
                updated_line = re.sub(rf'\b{name}\b', identifier, updated_line)
            updated_lines.append(updated_line)
        
        # Join the updated lines back into the batch with line breaks
        updated_batch = "\n".join(updated_lines)
        updated_batches.append(updated_batch)
    
    return updated_batches, name_mapping

        
def annotate_speaker_roles(transcript_batch, client,openai_chat_model):
    additional_instructions = """
    Below is a transcript of a conversation between a doctor and a patient. Try to separate the sentences and annotate the roles of the speakers, either doctor or patient:
    """
    user_content = additional_instructions + '\n\n' + transcript_batch
    chat_completion = client.chat.completions.create(
        model=openai_chat_model,  # specify the correct model here
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with annotating the roles of speakers in the following transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    # Extract the response content
    return chat_completion.choices[0].message.content

def main_backup(transcript, openai_key,openai_api_version,openai_chat_endpoint,openai_chat_model):
    client = AzureOpenAI(
    api_key=openai_key,
    api_version=openai_api_version,
    azure_endpoint=openai_chat_endpoint
    )
    batches = truncate_transcript(transcript)
    annotated_batches = []
    for batch in batches:
        annotated_text = annotate_speaker_roles(batch, client,openai_chat_model)
        annotated_batches.append(annotated_text)
    # Combine all annotated batches into a single text
    full_annotated_transcript = "\n\n".join(annotated_batches)
    return full_annotated_transcript


def main(transcript, openai_key, openai_api_version, openai_chat_endpoint, openai_chat_model):
    client = AzureOpenAI(
        api_key=openai_key,
        api_version=openai_api_version,
        azure_endpoint=openai_chat_endpoint
    )
    
    # Step 1: Split transcript into batches
    initial_batches = truncate_transcript(transcript)
    
    # Step 2: Perform name de-identification on each batch
    deidentified_batches, name_mapping = replace_names_with_identifiers(initial_batches, client, openai_chat_model)
    full_deidentified_text = "\n\n".join(deidentified_batches)

    # Step 3: Annotate speaker roles for each de-identified batch
    re_batched_text = truncate_transcript(full_deidentified_text)

    annotated_batches = []
    for batch in re_batched_text:
        annotated_text = annotate_speaker_roles(batch, client, openai_chat_model)
        annotated_batches.append(annotated_text)
    
    # Step 4: Combine the annotated and de-identified batches into a single text
    final_transcript = "\n\n".join(annotated_batches)
    
    return final_transcript

def save_annotated_transcript(annotated_transcript, filename, output_v1_path):
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_roles{ext}"
    output_path = os.path.join(output_v1_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(annotated_transcript)
    print(f"Annotated transcript saved to {output_path}")

# Set paths for input and output directories
#input_path = r'U:\FSDesktop\Documents\LLM_evaluation\Deidentified_Transcription_by_LLM\Transcript'
#output_path = r'U:\FSDesktop\Documents\LLM_evaluation\Deidentified_Transcription_by_LLM\gpt_note'

input_path = r'U:/FSDesktop/Documents/LLM_evaluation/transcripts/'
batch_path = r'U:/FSDesktop/Documents/LLM_evaluation/transcripts_batches/'
de_identify_path= r'U:/FSDesktop/Documents/LLM_evaluation/transcripts_deidentify/'
output_v1_path = r'U:/FSDesktop/Documents/LLM_evaluation/transcripts_roles/'

# List all files in the input directory
files = os.listdir(input_path)
#index = files.index(filename)
for filename in files:
    print(f"Processing file: {filename}")
    # Construct the full input file path
    file_path = os.path.join(input_path, filename)
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    # Annotate the transcript
    annotated_transcript = main(transcript, openai_key, openai_api_version, openai_chat_endpoint, openai_chat_model)
    save_annotated_transcript(annotated_transcript, filename, output_v1_path)
       

 
