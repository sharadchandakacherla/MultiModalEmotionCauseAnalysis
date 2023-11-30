import json
import os

import torch
from transformers import GPT2ForQuestionAnswering, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from tqdm import tqdm

from data.dataset_utils import read_examples, convert_examples_to_features

max_seq_length=512
doc_stride=128
max_query_length=64
train_batch_size=32
num_epochs = 50
n_gpu = torch.cuda.device_count()
do_eval = True
eval_metric = 'best_f1'
learning_rate=2e-05
eval_per_epoch=10
output_dir = f"{os.getcwd()}/metrics/"
train_batch_size=32
eval_batch_size=32
train_eval_ratio = 0.8

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # or another GPT-2 variant
model = GPT2ForQuestionAnswering.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

training_file = "/Users/sharadc/Documents/uic/semester4/CS598/MultiModalEmotionCauseAnalysis/data/raw/SemEval-2024_Task3/text/Subtask_1_2_train.json"
ip_data = []
with open(training_file, "r", encoding='utf-8') as reader:
    input_data = json.load(reader)

training_offset =  int(train_eval_ratio*len(input_data))
training_data = input_data[:training_offset]
eval_data = input_data[training_offset:]

examples_train = read_examples(training_data, is_training=True)
examples_eval = read_examples(eval_data, is_training=False)
train_features = convert_examples_to_features(
        examples=examples_train,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True)



print(f'here {type(train_features[0])}')

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                           all_start_positions, all_end_positions)
train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
train_batches = [batch for batch in train_dataloader]

print('processing eval data')
eval_features = convert_examples_to_features(
        examples=examples_eval,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
train_batches = [batch_t for batch_t in train_dataloader]
eval_dataloader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=True)


# Create DataLoader for training
# Fine-tuning settings
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3  # You may need to adjust this based on your dataset size and training needs

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):

        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        input_dict = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'start_positions': start_positions,
            'end_positions': end_positions
        }
        optimizer.zero_grad()
        outputs = model(**input_dict)
        loss = outputs.loss
        print(f'loss {loss}')
        loss.backward()

        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2_squad_model")
tokenizer.save_pretrained("fine_tuned_gpt2_squad_model")
