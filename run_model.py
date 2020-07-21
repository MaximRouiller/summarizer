from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from time import time

torch_device = 'cpu'
device = torch.device(torch_device)

print(f'Using "{torch_device}" for inferencing.')

content = """

"""

print('Loading model...')
start = time()
tokenizer = BartTokenizer.from_pretrained("./model/")

model = BartForConditionalGeneration.from_pretrained("./model/").to(device)
print(f'Model loaded in {round(time()-start, 2)}s.')

print('Tokenizing data...')
start = time()
article_input_ids = tokenizer.batch_encode_plus([content.replace('\n','')], return_tensors='pt', max_length=1024)['input_ids'].to(device)
print(f'Data tokenized in {round(time()-start, 2)}s.')

print('Executing model...')
start = time()
summary_ids = model.generate(article_input_ids,
                                num_beams = 4,
                                length_penalty=2.0,
                                max_length=142,
                                min_len=56,
                                no_repeat_ngram_size=3)
print(f'Model executed in {round(time()-start, 2)}s.')

print('Generating result...')
start = time()
summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
print(f'Result generated in {round(time()-start, 2)}s.')


print(summary_txt)