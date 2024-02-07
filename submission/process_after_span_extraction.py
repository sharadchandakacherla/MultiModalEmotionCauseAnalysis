import argparse
import json
import os

def find_indices_for_prediction(pred_utt, conversations):
    # step 1 check for complete presence
    utts, speakers = zip(*[(x['text'], x['speaker']) for x in conversations])
    utts = list(utts)
    speakers = set(speakers)
    pairs = []
    for i, utt in enumerate(utts):
        if pred_utt in utt:
            # need to handle -1? is not found? not possible
            start_index = utt.find(pred_utt)
            end_index = start_index + len(pred_utt)
            utt_id = i + 1
            pairs.append([f'{utt_id}_{start_index}_{end_index}'])
            if start_index < 0 or end_index < 0:
                print("error")
            break

    speaker_counter = 0
    speakers_in_pred_utt = []
    for speaker in speakers:
        if f'{speaker}:' in pred_utt:
            speaker_counter += 1
            speakers_in_pred_utt.append(f'{speaker}:')

    for x in speakers_in_pred_utt:
        if x in pred_utt:
            pred_utt = pred_utt.replace(x, "TOKEN_SPEAKER:")

    if "TOKEN_SPEAKER" in pred_utt:
        split_preds = pred_utt.split('TOKEN_SPEAKER:')
        split_preds = [x.strip() for x in split_preds]
        for sp in split_preds:
            if sp != "":
                for i, utt in enumerate(utts):
                    if sp in utt:
                        # need to handle -1? is not found? not possible
                        start_index = utt.find(sp)
                        end_index = start_index + len(sp)
                        utt_id = i + 1
                        pairs.append([f'{utt_id}_{start_index}_{end_index}'])
                        if start_index < 0 or end_index < 0:
                            print("error")
                        break

    if pairs == []:
        print("error, empty pairs")
    pairs = [z for x in pairs for z in x]
    return pairs

# credit : adopted from SemEval 2024 authors evaluation.py
def convert_list_to_dict(data_list, main_key=''):
    new_dict = {}
    for x in data_list:
        if 'ID' in main_key:
            key = int(x[main_key])
        else:
            key = x[main_key]
        if key not in new_dict:
            new_dict[key] = x
        else:
            import sys
            sys.exit('Instance repeat!')
    return new_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide file path of the enriched spans of final results. All paths are absolute paths')
    parser.add_argument('--enriched_dataset_with_emotions', type=str, required=True, help="json file that contains evaluation data enriched with emotion for each utterances after emotion classification")
    parser.add_argument('--span_predictions_eval', type=str, required=True)
    parser.add_argument('--evaluation_span_extraction_file', type=str, required=True)
    parser.add_argument('--evaluation_dataset_file', type=str, required=True)
    parser.add_argument('--final_save_file', type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.enriched_dataset_with_emotions), f'Path {args.enriched_dataset_with_emotions} does not exist.'
    assert os.path.exists(args.span_predictions_eval), f'Path {args.span_predictions_eval} does not exist.'
    assert os.path.exists(args.evaluation_span_extraction_file), f'Path {args.evaluation_span_extraction_file} does not exist.'
    assert os.path.exists(args.evaluation_dataset_file), f'Path {args.evaluation_dataset_file} does not exist.'

    #this file is generated after the emotion classification
    with open(args.enriched_dataset_with_emotions) as f:
        enriched_data = json.load(f)

    #this file is from SpanBERT's output folder
    with open(args.span_predictions_eval) as f:
        eval_ = json.load(f)
    # print(len(eval_))

    #this file is the input eval dataset in SQuAD 2.0 format to the span extractor
    with open(args.evaluation_span_extraction_file) as f:
        eval_from_local = json.load(f)['data'][0]['paragraphs']

    # this file is the SemEval file
    with open(args.evaluation_dataset_file, "r") as f:
        eval_data = json.load(f)

    print(f'len of data {len(eval_data)}')
    eval_ds = convert_list_to_dict(eval_data, main_key="conversation_ID")

    not_counter = []
    counter = 0
    neg = 0
    total_unknowns = 0
    not_total_unknowns = 0
    utt_2_pairs = {}

    for qa in eval_from_local:
        data_from_eval = qa['qas'][0]

        qid = data_from_eval['id']
        full_qid = qid
        # qid = int(int(qid)/100)
        qid = f'{qid}'
        conv_id = int(int(qid) / 10000)
        utt_id = int(int(qid) % 10000)
        if conv_id == 1375:
            print("")
        eval_val = eval_[qid] if qid in eval_ else ""

        if eval_val != "":
            conversations = eval_ds[conv_id]['conversation']
            pairs = find_indices_for_prediction(eval_val, conversations)
            if f'{conv_id}_{utt_id}' in utt_2_pairs:
                utt_2_pairs[f'{conv_id}_{utt_id}'].extend(pairs)
            else:
                utt_2_pairs[f'{conv_id}_{utt_id}'] = pairs
        else:
            print("answer not found")

        for data in enriched_data:
            ec_pairs = data['emotion-cause_pairs']
            convid = data['conversation_ID']
            newlist_ecp = []
            for ec in ec_pairs:
                uttid, pred_emo = ec[0].split("_")
                if pred_emo != "neutral":
                    key = f'{convid}_{uttid}'
                    if key in utt_2_pairs:
                        for i in utt_2_pairs[key]:
                            newlist_ecp_el = [ec[0], i]
                            newlist_ecp.append(newlist_ecp_el)

            data['emotion-cause_pairs_new'] = newlist_ecp

    # deleting the redundant labels
    for i in enriched_data:
        i['emotion-cause_pairs'] = i['emotion-cause_pairs_new']
        del i['emotion-cause_pairs_new']

    # print("done")
    # writing the final file for prediction
    with open(args.final_save_file,'w') as f:
        json.dump(enriched_data, f, indent=4)
