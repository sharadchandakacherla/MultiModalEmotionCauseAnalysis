<h1> Methodology for Subtask 1 </h1>

<h1> Requirements </h1>
<ol>
<li> Create two environments (recommended, might be able to run only with step 2) one for Emotion classification and the other for Span extraction to avoid any conflicts amongst the dependencies</li>
<li> For the step 1 i.e, emotion extraction from the emotion_extraction.ipynb, the environment can be created by running the first cell of the notebook. <br/>These packages are :

```bash
pip install transformers[torch]
pip install datasets
pip install accelerate -U
pip install ipywidgets
pip install scikit-learn
pip install boto3
```

</li>
  <li>For the step 2, please refer to the file requirements.txt</li>  
<li>One can choose to install all the requirements either in a conda environment/ venv or directly on their python installation</li>

  <li>Install the file using the following command 

```bash 
pip install -r requirements.txt
```
</li>
<li>
All the weights and raw files used for the inference on trained models for the contest can be found in this 

[drive](https://drive.google.com/drive/folders/14kVF4-6L0wu5SGkPyUQTS8PCm0uuNjyO?usp=drive_link) link.
</li>
</ol>

<h1> Methodology </h1>
<p> Our methodology is very simple and can be seen in two sequential steps. It relies on our empirical finding based on the training set that : given the exact emotion expressed in the utterance we achieve high exact match causal span extraction using SpanBERT fine-tuned in a Question answering paradigm.
<ol>
  <li>Run the notebook <code>./emotion_extraction.ipynb</code> to extract emotions, and enrich the emotional utterances with emotion. Select the dataset and model save paths accordingly and edit Section 1.1 in the notebook. <b>Section 1.5 for training and evaluation on the Subtask_1_train.json file, Section 1.6 for evaluating on the Subtask_1_test.json file</b> </li>
  <li>Each input to our RoBERTa model looks like <code>U_i [SEP] U_all</code> , where U_i means the current utterance and U_all is all the conversations concatenated .</li> 
  Create the enriched files by using <b>Section 1.6 and onwards</b> from <code>./emotion_extraction.ipynb</code>. This will create the <code>enriched_data.json</code> <b>in the same directory as the saved models set in step 1</b>. Use the enriched files to further fine-tune FAIR's SpanBERT pre-trained on SQuAD 2.0 dataset with the given hyper-parameters with the following input / hard prompt in the QA form. <br />(During inference for the span extraction step, enriched_data available from the drive link <code>./semeval_submission/raw_files_used_for_codalab_leaderboard/enriched_data.json</code>) </li>
  <li><p>Each input prompt (hard prompt) to our SpanBERT model looks like a typical SQuAD input : <code>The current utterance is - {current_utterance}. What caused the {current_emotion} in the current utterance?[SEP] all_conversations_in_utterance_concatenated</code> </p></li>
  <li> Download FAIR's model finetuned on SQuAD 2.0 and place it in a directory <b>(only if training the model is required, for inference skip to step 7)</b>. To download the file run 

```bash 
bash ./SpanBERT/code/download_finetuned.sh fair_squad squad2  
``` 
</li>
  <li> <b>Use this command for running the span extraction code training (skip to next step if running inference on submitted model)</b> 

```bash
python ./SpanBERT/code/run_meca_final.py --do_train --do_eval   --model fair_squad/squad2 --train_file Subtask_1_train.json --train_batch_size 12 --eval_batch_size 12 --learning_rate 2e-5 --num_train_epochs 5 --max_seq_length 400 --doc_stride 128 --eval_metric f1 --output_dir meca_final_train
```
</li>
<li> <b>Use this commad for running the span extraction code inference </b> 

```bash
python ./SpanBERT/code/run_meca_final.py --do_eval --do_test --model span_model_rc --train_file ./raw_files_used_for_codalab_leaderboard/enriched_data.json --train_batch_size 12 --eval_batch_size 12 --learning_rate 2e-5 --num_train_epochs 5 --max_seq_length 400 --doc_stride 128 --eval_metric f1 --output_dir meca_final_eval_submission
```

</li>
<li> The last step will create two json files (a) SpanBERT predictions called <b>predictions_eval.json</b> (b) evaluation data in SQuAD 2.0 format called <b>test_set_squad_format.json</b> The second file will be used while reproducing the submission score</li>
<li> Now, run these <b>two scripts (a) and (b)</b> in order to get the data in the final format as required by SemEval organizers. <br />
<b>(a)</b> 

```bash
python process_after_span_extraction.py --enriched_dataset_with_emotion ./raw_files_used_for_codalab_leaderboard/enriched_data.json 
--span_predictions_eval ./SpanBERT/code/reduced_contexts/predictions_eval.json
--evaluation_span_extraction_file ./raw_files_used_for_codalab_leaderboard/semeval_test_set_from_remote_no_labels_regen.json
--evaluation_dataset_file Subtask_1_test.json
--final_save_file final_results.json
```

<b>(b)</b> 

```bash
python char_spans_to_token_spans.py --predicted_spans_path final_results.json
```
</li>
</ol>

[//]: # (<h1> Generate the output file to submit to codalab </h1>)

<h1> Credits </h1>
We are grateful for FAIR to have made the code for SpanBERT and the model weights publicly available for research.
Please refer to the original codebase here as written by Mandar Joshi et. al.

```
  @article{joshi2019spanbert,
      title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
      author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
      journal={arXiv preprint arXiv:1907.10529},
      year={2019}
    }
```
