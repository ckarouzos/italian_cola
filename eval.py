import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


dataset = load_dataset('json', data_files="./datasets/multilingual-alpaca-52k/it.json")
model = AutoModelForSequenceClassification.from_pretrained("iproskurina/tda-itabert-ita-cola")
tokenizer = AutoTokenizer.from_pretrained("iproskurina/tda-itabert-ita-cola")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

LLM = pipeline('text-generation', model='gpt2', device=device)


def get_responses(batch):
    texts = [instruction + " " + input for instruction, input in zip(batch['instruction'], batch['input'])]
    results = LLM(texts, **tokenizer_kwargs)
    responses = [result[0]['generated_text'] for result in results]
    batch['LLM_respose'] = responses
    return batch

def get_predictions(batch):
    texts = batch['output']
    results = classifier(texts, **tokenizer_kwargs)
    labels = [result['label'] for result in results]
    batch['predictions'] = labels
    return batch

tokenizer_kwargs = {'max_length':512, 'pad_token_id':50256, 'return_full_text': False}
dataset['train'] = dataset['train'].map(get_responses, batched=True, batch_size=16)
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
predicted_dataset = dataset['train'].map(get_predictions, batched=True, batch_size=16)

acceptable_instruction_count = 0
acceptable_input_count = 0
aceptable_output_count = 0
count = 0
for example in tqdm(predicted_dataset):
    count += 1
    if example['predictions'] == 'Acceptable':
        acceptable_instruction_count += 1
    else:
        print(example['output'])
        print("\n")
print(f"Acceptable instruction: {acceptable_instruction_count / count}")
#print(f"Acceptable input: {acceptable_input_count / count}")
#print(f"Acceptable output: {aceptable_output_count / count}")

# 
# Acceptable instruction: 0.9449801913228332 (original dataset)