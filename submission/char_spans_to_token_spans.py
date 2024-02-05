import argparse
import json
import os
from copy import deepcopy

punctuations = {',', '!', '?', '.', ';', '$', '&', '"', '...'}


def map_to_token_spans(data_point: dict):
    emotion_cause_pairs = data_point['emotion-cause_pairs']
    conversation = data_point['conversation']

    for idx in range(len(emotion_cause_pairs)):
        _, cause_idx = emotion_cause_pairs[idx]
        utt_idx, start, end = [int(item) for item in cause_idx.split('_')]
        utt = conversation[utt_idx - 1]['text']
        pred_span_txt = utt[start: end]
        utt = utt.split()

        start_token_idx, end_token_idx = None, None

        pred_span = pred_span_txt.split()

        # n = len(utt)
        m = len(pred_span)

        for i in range(len(utt)):
            if ' '.join(utt[i: i + m]) == pred_span_txt:
                start_token_idx = i
                end_token_idx = i + m
                break

        if pred_span[0] in punctuations:
            start_token_idx += 1
        if pred_span[-1] in punctuations:
            end_token_idx -= 1

        emotion_cause_pairs[idx] = _, f'{utt_idx}_{start_token_idx}_{end_token_idx}'


parser = argparse.ArgumentParser(description='Provide file path of the predicted spans of final results.')
parser.add_argument('--predicted_spans_path', type=str, required=True)


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.exists(args.predicted_spans_path), f'Path {args.predicted_spans_path} does not exists.'

    base_path, file_name = os.path.split(args.predicted_spans_path)

    with open(os.path.join(base_path, file_name)) as f:
        data = json.load(f)

    result = deepcopy(data)

    for idx in range(len(result)):
        map_to_token_spans(result[idx])

    with open(os.path.join(base_path, f"token_{file_name}"), mode='w') as f:
        json.dump(result, fp=f, indent=4)
