# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

# from dataclasses import dataclass
# from typing import List

# from evaluate import evaluate_1_2


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _get_emotion_cause_labels(example, current_emotion, current_conv_id, conversation):
    # if conversation =="" or conversation == None:
    #     print(f"empty conv {example}")
    labels_in_conversation = example['emotion-cause_pairs']
    selected_labels = []
    for i in labels_in_conversation:
        if i[0] == f'{current_conv_id}_{current_emotion}':
            selected_labels.append(i[1])
    possible_cause_labels = []
    for j in selected_labels:
        causelabel = j.split("_")
        substring = causelabel[1]
        larger_string = conversation
        start = larger_string.find(substring)
        if start != -1:
            end = start + len(substring)
            possible_cause_labels.append([start, end, causelabel[1]])
        else:
            print(f'conv {example["conversation_ID"]}, label {j} has a case of anticipation')
            # print(f'start {start} ,ls: {larger_string} ,ss: {substring} ,conversation: {conversation}')
            # raise Exception(f"some issue in finding label {substring} in {example['conversation_ID']}")

    return possible_cause_labels
    
def read_squad_examplesV2(input_file, is_training=True):
    ds = []
    import json
    examples = input_file
    input_data = examples
    # with open(input_file, "r", encoding='utf-8') as reader:
    #     input_data = json.load(reader)
    print(f'no of examples : {len(input_data)}')

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }
    for example in input_data:
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation'][:i] if i > 0 else []

            conversation_until_now_arr = []
            if previous_conversations:

                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in conversation_until_now:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        if doc_tokens:
                            doc_tokens[-1] += c
                        else:
                            doc_tokens.append(c)
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            # sample = f'[CLS]{conversation_until_now}[CURRENT_UTT]{current_utterance}[SEP]What caused the emotion of {current_emotion} in the current utterance?'
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            # sample2 = f'{conversation_until_now}[SEP]{question_prompt}'
            # print(self.tokenizer(sample2))
            # print(sample2)
            if is_training:
                labels_arr = []
                labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                       current_conv_id=current_utterance_id,
                                                       conversation=conversation_until_now)
                # start_position, end_position = _get_emotion_cause_labels(example, current_emotion ,current_utterance, i)
                for labels in labels_arr:
                    start_offset = labels[0]
                    end_offset = labels[1]

                    start_position = char_to_word_offset[start_offset]
                    end_position = char_to_word_offset[end_offset - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    orig_answer_text = labels[2]
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        print("Could not find answer: '%s' vs. '%s'",
                              actual_text, cleaned_answer_text)
                        raise Exception(f"Error reading example {current_utterance_id}")

                    qexample = SquadExample(
                        qas_id=f"{example['conversation_ID']}_{i}",
                        question_text=question_prompt,
                        doc_tokens=doc_tokens,
                        orig_answer_text=conversation_until_now,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=False)
                    ds.append(qexample)
            else:
                qexample = SquadExample(
                    qas_id=f"{example['conversation_ID']}_{i}",
                    question_text=question_prompt,
                    doc_tokens=doc_tokens,
                    orig_answer_text=conversation_until_now,
                    start_position=-1,
                    end_position=-1,
                    is_impossible=False)
                ds.append(qexample)

    return ds

def convert_to_squadV2(input_data):
    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }

    data = []
    data_obj = {}
    data_obj['title'] = "SemEval-3"
    data_obj['paragraphs'] = []
    for example in input_data:
        idx_prefix = int(example['conversation_ID']) * 10000
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation'][:i] if i > 0 else []
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            conversation_until_now_arr = []
            if previous_conversations:
                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)
            labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                   current_conv_id=current_utterance_id,
                                                   conversation=conversation_until_now)
            for label_entry in labels_arr:
                obj = {}
                obj['context'] = conversation_until_now
                qas = {}
                qas['question'] = question_prompt
                qas['id'] = idx_prefix + current_utterance_id
                qas['is_impossibe'] = False
                qas['answers'] = [{"text": label_entry[2], "answer_start": label_entry[0]}]
                obj['qas'] = [qas]
                data_obj['paragraphs'].append(obj)
    data.append(data_obj)

    return {"data": data}

def convert_to_squadV3_full_contexts(input_data):
    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }

    data = []
    data_obj = {}
    data_obj['title'] = "SemEval-3"
    data_obj['paragraphs'] = []
    for example in input_data:
        idx_prefix = int(example['conversation_ID']) * 10000
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation']
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            conversation_until_now_arr = []
            if previous_conversations:
                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{txt}')
                # assumming full-stops exist after each utterance
            #             conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " . ".join(conversation_until_now_arr)
            labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                   current_conv_id=current_utterance_id,
                                                   conversation=conversation_until_now)
            for le_idx, label_entry in enumerate(labels_arr):
                obj = {}
                obj['context'] = conversation_until_now
                qas = {}
                qas['question'] = question_prompt
                # qas['id'] = idx_prefix + current_utterance_id
                qas['id'] = (idx_prefix + current_utterance_id)*100 + le_idx
                qas['is_impossibe'] = False
                qas['answers'] = [{"text": label_entry[2], "answer_start": label_entry[0]}]
                obj['qas'] = [qas]
                data_obj['paragraphs'].append(obj)
    data.append(data_obj)

    return {"data": data}

def convert_to_squadV3_full_contexts_improvised(input_data, train=False):
    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }


    def get_emotion(utt, emotions_preds):
        for emo in emotions_preds:
            utt_id_, utt_emo = emo[0].split("_")
            if utt == int(utt_id_):
                return utt_emo

    data = []
    data_obj = {}
    data_obj['title'] = "SemEval-3"
    data_obj['paragraphs'] = []
    for example in input_data:
        idx_prefix = int(example['conversation_ID']) * 10000
        ecp = example['emotion-cause_pairs']
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = get_emotion(current_utt['utterance_ID'], ecp)
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation']
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            conversation_until_now_arr = []
            if previous_conversations:
                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            #             conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)

            if train:

                labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                       current_conv_id=current_utterance_id,
                                                       conversation=conversation_until_now)
                for le_idx, label_entry in enumerate(labels_arr):
                    obj = {}
                    obj['context'] = conversation_until_now
                    qas = {}
                    qas['question'] = question_prompt
                    # qas['id'] = idx_prefix + current_utterance_id
                    qas['id'] = (idx_prefix + current_utterance_id)*100 + le_idx
                    qas['is_impossibe'] = False
                    qas['answers'] = [{"text": label_entry[2], "answer_start": label_entry[0]}]
                    obj['qas'] = [qas]
                    data_obj['paragraphs'].append(obj)
            else:
                obj = {}
                obj['context'] = conversation_until_now
                qas = {}
                qas['question'] = question_prompt
                qas['id'] = idx_prefix + current_utterance_id
                # qas['id'] = (idx_prefix + current_utterance_id) * 100 + le_idx
                qas['is_impossibe'] = False
                obj['qas'] = [qas]
                data_obj['paragraphs'].append(obj)

    data.append(data_obj)

    return {"data": data}

def does_anticipation_exist_in_text(point, id):
    for entry in point['emotion-cause_pairs']:
        id1 = int(entry[0].split("_")[0])
        if id == id1:
            id2 = int(entry[1].split("_")[0])
            if id2 > id1:
                return True
    return False

def convert_to_squadV4_full_contexts_for_anticipation(input_data):
    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }

    data = []
    data_obj = {}
    data_obj['title'] = "SemEval-3"
    data_obj['paragraphs'] = []
    anticipation = 0
    for example in input_data:
        idx_prefix = int(example['conversation_ID']) * 10000
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            # check if anticipation exists for this conversation, if yes send full context, else proceed as is

            if does_anticipation_exist_in_text(example, i + 1):
                previous_conversations = example['conversation']
            else:
                previous_conversations = example['conversation'][:i + 1]

            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            conversation_until_now_arr = []
            if previous_conversations:
                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            # conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)
            labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                   current_conv_id=current_utterance_id,
                                                   conversation=conversation_until_now)
            for label_entry in labels_arr:
                obj = {}
                obj['context'] = conversation_until_now
                qas = {}
                qas['question'] = question_prompt
                qas['id'] = idx_prefix + current_utterance_id
                qas['is_impossibe'] = False
                qas['answers'] = [{"text": label_entry[2], "answer_start": label_entry[0]}]
                obj['qas'] = [qas]
                data_obj['paragraphs'].append(obj)
    data.append(data_obj)

    return {"data": data}

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging,
                     version_2_with_negative):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
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
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
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
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
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
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

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

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

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
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


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


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    if normalize_answer(a_gold) == normalize_answer(a_pred):
        printer_match = {"a_gold": a_gold, "predicted": a_pred}
        logger.info(f"did match, {printer_match}")
        logger.info(f"did match - {a_pred}")
    else:
        printer = {"a_gold": a_gold, "predicted": a_pred}
        logger.info(f"didn't match, {printer}")

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# def find_closest_answers(dataset, preds):
#     answers = {}
#     for article in dataset:
#         for para in article['paragraphs']:
#             for question_answer in para['qas']:
#                 question_id = question_answer['id']
#                 gold_answers = [a['text'] for a in question_answer['answers'] if normalize_answer(a['text'])]
#                 if not gold_answers:
#                     gold_answers = ['']
#                 if question_id not in preds:
#                     continue
#                 answer_pred = preds[question_id]
#                 answers[question_id] = answer_pred if normalize_answer(a_gold) == normalize_answer(answer_pred) else


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers'] if normalize_answer(a['text'])]
                if not gold_answers:
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                logger.info(f'question qid {qid} - {qa}')
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False, og_val_dataset=None):
    all_results = []
    model.eval()

    if og_val_dataset is None:
        og_val_dataset = eval_dataset

    for idx, (input_ids, input_mask, segment_ids, example_indices) in enumerate(eval_dataloader):
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    preds, nbest_preds, na_probs = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging,
                         args.version_2_with_negative)

    if pred_only:
        if args.version_2_with_negative:
            for k in preds:
                if na_probs[k] > na_prob_thresh:
                    preds[k] = ''
        return {}, preds, nbest_preds

    if args.version_2_with_negative:
        qid_to_has_ans = make_qid_to_has_ans(eval_dataset)
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        result = make_eval_dict(exact_thresh, f1_thresh)
        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(result, has_ans_eval, 'HasAns')
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(result, no_ans_eval, 'NoAns')
        find_all_best_thresh(result, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
        for k in preds:
            if na_probs[k] > result['best_f1_thresh']:
                preds[k] = ''
    else:
        #commenting next line for testing 
#        c exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        # f"{conversation_ID}_{i}"  ; i happens to align with utterance_ID
        #         question_ids = list(preds.keys())
        #         datapoints = []

        #         for datapoint in og_val_dataset:
        #             conversation_id = datapoint['conversation_ID']
        #             conversations = []
        #             emotion_cause_pairs = []

        #             for conversation in datapoint['conversation']:
        #                 conversations.append(ConversationItem(conversation['utterance_ID'], conversation['text'], conversation['speaker']))

        #             for pair in datapoint['emotion_cause_pairs']:
        #                 emotion_cause_pairs.append(EmotionCausePairItem(*pair))

        #             datapoints.append(DataPoint(conversation_id, conversations, emotion_cause_pairs))

        #         pred_data = [data.to_dict() for data in datapoints]
        #         evaluate_1_2(pred_data=pred_data, gold_data=og_val_dataset)

        # TODO - Call semeval evaluate methods here...
#       c  result = make_eval_dict(exact_raw, f1_raw)
#     c logger.info(f'here results {result}')
        logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
#    c return result, preds, nbest_preds
    return {}, preds, nbest_preds

def get_indices(main_string, substring):
    # print("inside get_indices")
    indices = []
    start_index = 0
    # print(f'ms : {main_string}')
    # print(f'ss : {substring}')
    while True:
        index = main_string.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    # print(indices)
    if len(indices) == 0:
        raise Exception("substring not found")
    return indices[0]


def add_span_indices(data_point):
    nbest_answers = data_point['answers']
    for asp in data_point['labelled_answer_spans']:
        aspect = data_point['labelled_answer_spans'][asp]
        # print(f'setting indices for {asp}')
        for x in aspect:
            # print(x)
            for i, j in enumerate(nbest_answers):
                newline_removed_answer = j.replace("\n", "")
                span = x['txt'].strip()
                span = span[1:]  # remove : char in the beginning
                span = span.strip()
                x['mod_txt'] = span
                # print(f"newline_removed_answer {newline_removed_answer}")
                if span in newline_removed_answer:
                    # print(f"answer_{i} --> {x['txt']}")
                    answer = j
                    # print(f'here {answer}')
                    start_idx = get_indices(answer.replace("\n", ""), span)
                    # print(f'inices {start_idx}')
                    x['answer_index_start'] = start_idx
                    x['answer_index_end'] = start_idx + len(span)
                    x['original_answer_idx'] = i
                    x['original_answer'] = answer
                    break


def final_check(data_tf):
    counter = 0
    uris = []
    final_data = []
    for d_i, data_point in enumerate(data_tf):
        nbest_answers = data_point['answers']
        for asp in data_point['labelled_answer_spans']:
            aspect = data_point['labelled_answer_spans'][asp]
            # print(aspect)
            for x in aspect:

                if 'original_answer_idx' not in x:
                    if x['mod_txt'] in data_point['context']:
                        uris.append(data_point['uri'])
                    else:
                        uris.append(data_point['uri'])
                    counter += 1
                    break
                oa = x['original_answer_idx']
                answer_index_start = x['answer_index_start']
                answer_index_end = x['answer_index_end']
                if nbest_answers[oa].replace("\n", "")[answer_index_start:answer_index_end] != x['mod_txt']:
                    raise (f'{data_point["uri"]} has some problems with the span indices')
    print(counter)
    fout = set(uris)
    for i in data_tf:
        if i['uri'] in fout:
            continue
        final_data.append(i)

    return final_data

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained("spanbert-base-cased", do_lower_case=args.do_lower_case)

    # dataset_path = "/workspace/SpanBERT/code/redundant/MultiModalEmotionCauseAnalysis/v2/results_30_epochs_data_leak_corrected_shuffled_ui_uall_custom_roberta_base_weighted/enriched_data.json"
    dataset_path = args.train_file
    import json
    with open(dataset_path, "r") as f:
        data = json.load(f)
    print(f'len of data {len(data)}')

    data_tf = data

    train_eval_ratio = 0.8
    input_data = data_tf
    training_offset = int(train_eval_ratio * len(input_data))

    og_val_dataset = input_data[training_offset:]

    if args.do_train or (not args.eval_test):
        # with open(args.dev_file) as f:
        #     dataset_json = json.load(f)
#         eval_data = input_data[training_offset:]
        eval_data = input_data
        #         squad_ds_eval = convert_to_squadV2(eval_data)
        squad_ds_eval = convert_to_squadV3_full_contexts_improvised(eval_data, False)
        with open(os.path.join(args.output_dir, 'semeval_test_set_from_remote_no_labels_regen.json'), 'w') as es:
            json.dump(squad_ds_eval, es)
        eval_dataset = squad_ds_eval['data']
        # eval_dataset = eval_data
        eval_examples = read_squad_examples(input_file=eval_dataset, is_training=False, version_2_with_negative=args.version_2_with_negative)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:

        training_data = input_data[:training_offset]
        #         squad_ds = convert_to_squadV2(training_data)
        squad_ds = convert_to_squadV3_full_contexts_improvised(training_data, False)
        squad_ds = squad_ds['data']

        train_examples = read_squad_examples(
            input_file=squad_ds, is_training=True, version_2_with_negative=args.version_2_with_negative)

        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            model = BertForQuestionAnswering.from_pretrained(
                args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")
                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                           warmup_linear(global_step / num_train_optimization_steps,
                                                         args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                            epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

                        save_model = False
                        if args.do_eval:
                            # og_val_dataset = input_data[training_offset:]
                            result, _, _ = \
                                evaluate(args, model, device, eval_dataset,
                                         eval_dataloader, eval_examples, eval_features, og_val_dataset=og_val_dataset)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                        else:
                            save_model = True
                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

    if args.do_eval:
        if args.eval_test:
            #             with open(args.test_file) as f:
            #                 dataset_json = json.load(f)
            #             eval_dataset = dataset_json['data']
            #             eval_examples = read_squad_examples(
            #                 input_file=args.test_file, is_training=False, version_2_with_negative= False)
            eval_data = input_data[training_offset:]
            #             squad_ds_eval = convert_to_squadV2(eval_data)
            squad_ds_eval = convert_to_squadV3_full_contexts_improvised(eval_data,False)

            eval_dataset = squad_ds_eval['data']
            # eval_dataset = eval_data
            eval_examples = read_squad_examples(input_file=eval_dataset, is_training=False,
                                                version_2_with_negative=args.version_2_with_negative)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        model = BertForQuestionAnswering.from_pretrained(args.model)
        if args.fp16:
            model.half()
        model.to(device)

        na_prob_thresh = 1.0
        if args.version_2_with_negative:
            eval_result_file = os.path.join(args.output_dir, "eval_results.txt")
            if os.path.isfile(eval_result_file):
                with open(eval_result_file) as f:
                    for line in f.readlines():
                        if line.startswith('best_f1_thresh'):
                            na_prob_thresh = float(line.strip().split()[-1])
                            logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        result, preds, nbest_preds = \
            evaluate(args, model, device, eval_dataset,
                     eval_dataloader, eval_examples, eval_features,
                     na_prob_thresh=na_prob_thresh,
                     pred_only=args.eval_test, og_val_dataset=og_val_dataset)
        with open(os.path.join(args.output_dir, "predictions_eval.json"), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, "n_best_predictions_eval.json"), "w") as writer:
            writer.write(json.dumps(nbest_preds, indent=4) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--dev_file", default=None, type=str,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                             "how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1', type=str)
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. "
                             "This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    args = parser.parse_args()

    main(args)