import collections
import math
import string
import time
import re
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, get_scheduler
import os
import json

from data.dataset_utils import read_examples, convert_examples_to_features

model_name = 'SpanBERT/spanbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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


# training
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * num_epochs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#trainer
# Define training arguments
# learning_rate = 2e-5
# num_epochs = 3
# batch_size = 16
# weight_decay = 0.01
# Create a PyTorch DataLoader
train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
train_batches = [batch_t for batch_t in train_dataloader]
eval_dataloader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=True)
#eval_batches = [batch_e for batch_e in eval_dataloader]
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


eval_step = max(1, len(train_batches) // eval_per_epoch)
best_result = None
lrs = [learning_rate] if learning_rate else \
    [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
for lr in lrs:
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    global_step = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dataloader), colour='cyan', leave=True) as bar:
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                input_dict = {
                    'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': segment_ids,
                    'start_positions': start_positions,
                    'end_positions': end_positions
                }
                output = model(**input_dict)

                start_logits = output['start_logits']
                end_logits = output['end_logits']
                loss = output.loss
                if n_gpu > 1:
                    loss = loss.mean()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                #if (step + 1) % eval_step == 0:
                print('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'
                      .format(epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

            # save_model = False


                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(output_dir, f"train/lr_{lr}_{int(time.time())}.bin")
                output_config_file = os.path.join(output_dir, f"train/lr_{lr}_{int(time.time())}.json")
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_dir)
                # if best_result:
                #     with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                #         for key in sorted(best_result.keys()):
                #             writer.write("%s = %s\n" % (key, str(best_result[key])))
            bar.update()
            bar.set_description(f'Training {epoch}/{num_epochs} - Loss {tr_loss / nb_tr_steps:.3f}')


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Average training Loss: {avg_loss}")


        #evaluate

        with tqdm(total=len(eval_dataloader), colour='red', leave=True) as bar:
            for id,(input_ids, input_mask, segment_ids, start_positions, end_positions) in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids
                input_mask
                segment_ids
                start_positions, end_positions = batch
                input_dict = {
                    'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': segment_ids,
                    'start_positions': start_positions,
                    'end_positions': end_positions
                }
                output = model(**input_dict)
                start_logits = output['start_logits']
                end_logits = output['end_logits']
                start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

                max_start, max_end, max_prob = 0, 0, 0
                for it, k in enumerate(start_probs):
                    for i in range(len(k)):
                        for j in range(len(end_probs[it])):
                            if start_probs[it][i] * end_probs[it][j] > max_prob and i <= j:
                                max_start, max_end = i, j
                                max_prob = start_probs[it][i] * end_probs[it][j]

                    # answer = tokenizer.convert_tokens_to_string(
                    #     tokenizer.convert_ids_to_tokens(input_dict['input_ids'][it][max_start:max_end + 1]))
                answer = tokenizer.decode(input_dict['input_ids'][0][max_start:max_end + 1],skip_special_tokens=True)

# Save the fine-tuned model
    model.save_pretrained("fine_tuned_spanbert")










#----
start_probs, end_probs = None, None
#input = train_dataloader.dataset.__getitem__(0)
for step,ip in enumerate(train_batches):
    input_ids, input_mask, segment_ids, start_positions, end_positions = ip
    input_dict = {
        'input_ids': input_ids,
        'attention_mask': input_mask,
        'token_type_ids': segment_ids,
        'start_positions': start_positions,
        'end_positions': end_positions
    }
    #print(input_ids.shape)
    output = model(**input_dict)

    start_logits = output['start_logits']
    end_logits = output['end_logits']
    start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
    # Find the best answer span


