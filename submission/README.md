<h1> Methodology for Subtask 1 </h1>

<h1> Requirements </h1>
<ol>
  <li>One can choose to install all the requirements either in a conda environment/ venv or directly on their python installation</li>
  <li>please refer to the file requirements.txt</li>
  <li>Install the file using the following command <code>pip install -r requirements.txt</code></li>
</ol>

<h1> Methodology </h1>
<p> Our methodology is very simple and can be seen in two sequential steps. It relies on our empirical finding based on the training set that : given the exact emotion expressed in the utterance we achieve high exact match causal span extraction using SpanBERT fine-tuned in a Question answering paradigm.
<ol>
  <li>Run the notebook to extract emotions, and enrich the emotional utterances with emotion</li>
  <li><p>Each input to our RoBERTa model looks like U_i \<\SEP\> U_all , where U_i means the current utterance and U_all is all the conversations concatenated </p></li> 
  <li>Use the enriched files to further fine-tune FAIR's SpanBERT pre-trained on SQuAD 2.0 dataset with the given hyper-parameters with the following input / hard prompt in the QA form</li>
  <li><p>Each input to our SpanBERT model looks like a SQuAD input : "The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"[SEP] All converations concatednated </p></li>
  <li> Download FAIR's model finetuned on SQuAD 2.0 and place it in a directory. To download the file run <code>bash SpanBERT/code/download_finetuned.sh \<\model_dir\> squad2 </model_dir> </code> </li>
  <li> use this commad for running the span extraction code <code>python SpanBERT/code/run_meca.py --do_train --do_eval  --model /workspace/masumm_sb/SpanBERT/meca_output_fair_squad  --train_file Subtask_1_train.json   --train_batch_size 12  --eval_batch_size 12  
  --learning_rate 2e-5 --num_train_epochs 5 --max_seq_length 400 --doc_stride 128 --eval_metric f1 --output_dir meca_output_fair_squad_eval_2</code></li>
</ol>

<h1> Generate the output file to submit to codalab </h1>

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
