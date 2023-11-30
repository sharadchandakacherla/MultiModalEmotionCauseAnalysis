import collections
import math
import string
import time
import re
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
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

                # max_start_index = torch.argmax(start_logits)
                # max_end_index = torch.argmax(end_logits)
                #
                # # Extract the best answer span
                # best_answer_span = input_dict["input_ids"][0][max_start_index:max_end_index + 1]
                # best_answer_text = tokenizer.decode(best_answer_span)

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






RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
def evaluate(self, eval_data, output_dir, verbose_logging=False):
    """
    Evaluates the model on eval_data.

    Utility function to be used by the eval_model() method. Not intended to be used directly.
    """
    tokenizer = self.tokenizer
    device = self.device
    model = self.model
    args = self.args

    if isinstance(eval_data, str):
        with open(eval_data, "r", encoding=self.args.encoding) as f:
            eval_examples = json.load(f)
    else:
        eval_examples = eval_data

    eval_dataset, examples, features = self.load_and_cache_examples(
        eval_examples, evaluate=True, output_examples=True
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if self.args.fp16:
        from torch.cuda import amp

    all_results = []
    for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Evaluation", position=0, leave=True):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if self.args.model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "electra",
                "xlmroberta",
                "bart",
            ]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            if self.args.fp16:
                with amp.autocast():
                    outputs = model(**inputs)
                    eval_loss += outputs[0].mean().item()
            else:
                outputs = model(**inputs)
                eval_loss += outputs[0].mean().item()

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][i]),
                    end_logits=to_list(outputs[1][i]),
                )
                all_results.append(result)

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    prefix = "test"
    os.makedirs(output_dir, exist_ok=True)

    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

    all_predictions, all_nbest_json, scores_diff_json = write_predictions(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        True,
        args.null_score_diff_threshold,
    )

    return all_predictions, all_nbest_json, scores_diff_json, eval_loss

def predict(self, to_predict, n_best_size=None):
    """
    Performs predictions on a list of python dicts containing contexts and qas.

    Args:
        to_predict: A python list of python dicts containing contexts and questions to be sent to the model for prediction.
                    E.g: predict([
                        {
                            'context': "Some context as a demo",
                            'qas': [
                                {'id': '0', 'question': 'What is the context here?'},
                                {'id': '1', 'question': 'What is this for?'}
                            ]
                        }
                    ])
        n_best_size (Optional): Number of predictions to return. args.n_best_size will be used if not specified.

    Returns:
        list: A python list  of dicts containing the predicted answer/answers, and id for each question in to_predict.
        list: A python list  of dicts containing the predicted probability/probabilities, and id for each question in to_predict.
    """  # noqa: ignore flake8"
    tokenizer = self.tokenizer
    device = self.device
    model = self.model
    args = self.args

    if not n_best_size:
        n_best_size = args.n_best_size

    self._move_model_to_device()

    eval_examples = build_examples(to_predict)
    eval_dataset, examples, features = self.load_and_cache_examples(
        eval_examples, evaluate=True, output_examples=True, no_cache=True
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if self.args.fp16:
        from torch.cuda import amp

    all_results = []
    for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Prediction", position=0, leave=True):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if self.args.model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "electra",
                "xlmroberta",
                "bart",
            ]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            if self.args.fp16:
                with amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                result = RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][i]),
                    end_logits=to_list(outputs[1][i]),
                )
                all_results.append(result)

    if args.model_type in ["xlnet", "xlm"]:
        answers = get_best_predictions_extended(
            examples,
            features,
            all_results,
            n_best_size,
            args.max_answer_length,
            model.config.start_n_top,
            model.config.end_n_top,
            True,
            tokenizer,
            args.null_score_diff_threshold,
        )
    else:
        answers = get_best_predictions(
            examples, features, all_results, n_best_size, args.max_answer_length, False, False, True, False,
        )

    answer_list = [{"id": answer["id"], "answer": answer["answer"][:-1]} for answer in answers]
    probability_list = [{"id": answer["id"], "probability": answer["probability"][:-1]} for answer in answers]

    return answer_list, probability_list


def get_best_predictions_extended(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        start_n_top,
        end_n_top,
        version_2_with_negative,
        tokenizer,
        verbose_logging,
):
    """ XLNet write prediction logic (more complex than Bert's).
                    Write final predictions to the json file and log-odds of null if needed.
                    Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"],
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True,
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob,
                                 end_log_prob=pred.end_log_prob, )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

        all_best = [
            {
                "id": id,
                "answer": [answer["text"] for answer in answers],
                "probability": [answer["probability"] for answer in answers],
            }
            for id, answers in all_nbest_json.items()
        ]
    return all_best


def get_best_predictions(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True, )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit, ))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    all_best = [
        {
            "id": id,
            "answer": [answer["text"] for answer in answers],
            "probability": [answer["probability"] for answer in answers],
        }
        for id, answers in all_nbest_json.items()
    ]
    return all_best


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                print(
                    "Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text,
                )
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                print("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                print("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position: (orig_end_position + 1)]
        return output_text

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def write_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True,)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, all_nbest_json, scores_diff_json

#----
# start_probs, end_probs = None, None
# #input = train_dataloader.dataset.__getitem__(0)
# for step,ip in enumerate(train_batches):
#     input_ids, input_mask, segment_ids, start_positions, end_positions = ip
#     input_dict = {
#         'input_ids': input_ids,
#         'attention_mask': input_mask,
#         'token_type_ids': segment_ids,
#         'start_positions': start_positions,
#         'end_positions': end_positions
#     }
#     #print(input_ids.shape)
#     output = model(**input_dict)
# 
#     start_logits = output['start_logits']
#     end_logits = output['end_logits']
#     start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
#     end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
    # Find the best answer span


