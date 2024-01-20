import os
import random
import torch
import numpy as np
import json
from enum import IntEnum
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import TrainingType


class DatasetConfig(IntEnum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class EmotionCausalDataset(Dataset):
    """
    Dataset class for subtask 1
    """

    def __init__(self, path: str, config: DatasetConfig, training_type: TrainingType, tokenizer: AutoTokenizer,
                 device=torch.device('cpu'), seed=42, split=0.8, special_token='<SEP>'):
        self.path = path
        self.device = device
        self.tokenizer = tokenizer
        self._config = config
        self.seed = seed
        self.split = split
        self.training_type = training_type
        self.SPECIAL_TOKEN = special_token

        if self._config != DatasetConfig.TEST:
            with open(os.path.join(self.path, 'Subtask_1_train.json')) as f:
                data = json.load(f)

            sampling_idx = list(range(len(data)))
            random.seed(self.seed)
            random.shuffle(sampling_idx)
            ratio = int(split * len(data))
            sampling_idx = sampling_idx[: ratio] if self._config == DatasetConfig.TRAIN else sampling_idx[ratio:]
        else:
            # TODO: Mock on trial data
            data = []
            sampling_idx = list(range(len(data)))

        self._sampling_idx = sampling_idx
        self.data = data
        self.processed_data = []
        self.process_dataset()

    def process_dataset(self):
        processed_data = []

        if self.training_type == TrainingType.JOINT_TRAINING:
            processed_data = self._process_data_for_joint_model()
        elif self.training_type == TrainingType.EMOTION_CLASSIFICATION:
            processed_data = self._process_data_for_emotion_classification()
        elif self.training_type == TrainingType.SPAN_CLASSIFICATION:
            processed_data = self._process_data_for_span_classification()

        self.processed_data = processed_data

    def _process_data_for_span_classification(self):
        processed_data = []
        if self._config != DatasetConfig.TEST:
            for idx in self._sampling_idx:
                scene = self.data[idx]
                conversations = scene['conversation']

                for conv in conversations:
                    conv['caused_by'] = {}

                causal_pairs = scene['emotion-cause_pairs']

                for emotion_effect, emotion_cause in causal_pairs:
                    effect_idx = int(emotion_effect.split('_')[0]) - 1
                    cause_idx = int(emotion_cause.split('_')[0]) - 1
                    cause_span_txt = emotion_cause.split('_')[1]
                    conversations[effect_idx]['caused_by'][cause_idx] = cause_span_txt

                utt_all = ' '.join(conv['text'] for conv in conversations)
                # n^2 new examples for joint training.
                for i, conv_i in enumerate(conversations):
                    emotion = conv_i['emotion']
                    caused_in_i = conv_i['caused_by']
                    utt_i = conv_i["text"]
                    prompt = f"The current utterance is - {utt_i}. What caused the {emotion} in the current utterance?"
                    for j, conv_j in enumerate(conversations):
                        utt_j = conv_j['text']
                        prefix = f'{prompt} {self.SPECIAL_TOKEN}'
                        text = f'{prefix}{utt_j} {self.SPECIAL_TOKEN} {utt_all}'
                        spans = [-1, -1]

                        if j in caused_in_i:
                            cause_span_txt = caused_in_i[j]
                            idx = utt_j.find(cause_span_txt)
                            spans = [idx + len(prefix), idx + len(cause_span_txt) + len(prefix)]

                        processed_data.append({
                            'text': text,
                            'causal_span': spans,
                            'emotion': emotion
                        })

        else:
            raise NotImplementedError('Need to check in trail data.')

        return processed_data

    def _process_data_for_emotion_classification(self) -> List[Dict]:
        processed_data = []

        if self._config != DatasetConfig.TEST:
            for idx in self._sampling_idx:
                scene = self.data[idx]
                conversations = scene['conversation']
                causal_pairs = scene['emotion-cause_pairs']
                utt_all = ' '.join(conv['text'] for conv in conversations)

                for conv in conversations:
                    utt_i = conv['text']
                    emotion = conv['emotion']
                    text = f'{utt_i} {self.SPECIAL_TOKEN} {utt_all}'
                    processed_data.append({
                        'text': text,
                        'emotion': emotion
                    })

        return processed_data

    def _process_data_for_joint_model(self) -> List[Dict]:
        processed_data = []

        if self._config != DatasetConfig.TEST:
            for idx in self._sampling_idx:
                scene = self.data[idx]
                conversations = scene['conversation']

                for conv in conversations:
                    conv['caused_by'] = {}

                causal_pairs = scene['emotion-cause_pairs']

                for emotion_effect, emotion_cause in causal_pairs:
                    effect_idx = int(emotion_effect.split('_')[0]) - 1
                    cause_idx = int(emotion_cause.split('_')[0]) - 1
                    cause_span_txt = emotion_cause.split('_')[1]
                    conversations[effect_idx]['caused_by'][cause_idx] = cause_span_txt

                utt_all = ' '.join(conv['text'] for conv in conversations)

                # n^2 new examples for joint training.

                for i, conv_i in enumerate(conversations):
                    emotion = conv_i['emotion']
                    caused_in_i = conv_i['caused_by']
                    utt_i = conv_i["text"]
                    for j, conv_j in enumerate(conversations):
                        utt_j = conv_j['text']
                        text = f'{utt_i} {self.SPECIAL_TOKEN} {utt_j} {self.SPECIAL_TOKEN} {utt_all}'
                        spans = [-1, -1]

                        if j in caused_in_i:
                            cause_span_txt = caused_in_i[j]
                            idx = text.find(cause_span_txt)
                            spans = [idx, idx + len(cause_span_txt)]

                        processed_data.append({
                            'text': text,
                            'causal_span': spans,
                            'emotion': emotion
                        })

        else:
            raise NotImplementedError('Need to check in trail data.')

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        scene = self.processed_data[item]
        text_inp = scene['text']
        causal_span_label = scene.get('causal_span', None)
        emotion_label = scene.get('emotion', None)

        # TODO: Apply tokenizers, convert labels to tensors.

        if causal_span_label is None and emotion_label is None:
            return text_inp
        elif causal_span_label is None:
            return text_inp, emotion_label
        elif emotion_label is None:
            return text_inp, causal_span_label
        else:
            return text_inp, causal_span_label, emotion_label


if __name__ == "__main__":
    path = '../data/raw/SemEval-2024_Task3/dataset_final/train/text_files/text'
    dataset = EmotionCausalDataset(path=path, device="cpu", config=DatasetConfig.TRAIN,
                                   training_type=TrainingType.SPAN_CLASSIFICATION, tokenizer=None)
    print(len(dataset))
