import os
import json
from typing import Union, Type

import torch
from torch import multiprocessing as mp
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
from tqdm import tqdm

from config import TrainerConfig, TrainingType, TaskSolve
from dataset import DatasetConfig, EmotionCausalDataset
from models import ModelBaseClass, JointModel, EmotionClassification, SpanClassification
from metrics.evaluate import evaluate_runtime, evaluate_1_2
import torch.nn as nn


def select_model(training_type: TrainingType, solve_task: TaskSolve) -> Union[None, Type[ModelBaseClass]]:
    model = None

    if solve_task == TaskSolve.TASK1:
        if training_type == TrainingType.JOINT_TRAINING:
            model = JointModel
        elif training_type == TrainingType.EMOTION_CLASSIFICATION:
            model = EmotionClassification
        elif training_type == TrainingType.SPAN_CLASSIFICATION:
            model = SpanClassification
        else:
            raise NotImplementedError(f'ill-defined training type {training_type} for task type {solve_task}')
    elif solve_task == TaskSolve.TASK2:
        # TODO: Next phase.
        pass

    return model


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Trainer:
    def __init__(self, config: TrainerConfig, rank: int = 0, n_gpus: int = 0):
        self.config = config
        self.rank = rank
        self.n_gpus = n_gpus

        if self.rank == 0:
            os.makedirs(os.path.join(config.base_path, config.model_save_path), exist_ok=True)
            os.makedirs(os.path.join(config.base_path, config.model_log_path), exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            torch.cuda.empty_cache()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cpu_device = torch.device('cpu')

        model_cls = select_model(config.training_type, config.solve_task)

        self.model = model_cls(base_model_name=config.base_model_name, no_classes=config.no_classes)

        if config.solve_task == TaskSolve.TASK1:
            self.model.add_special_token_to_tokenizer(config.special_token)

        if config.freeze_base_model:
            self.model.freeze_base_model()

        if self.n_gpus > 1:
            setup(self.rank, self.n_gpus)

        if config.solve_task == TaskSolve.TASK1:
            path = os.path.join(config.base_path, 'data', 'text')
            tokenizer = self.model.tokenizer()
            train_dataset = EmotionCausalDataset(path, DatasetConfig.TRAIN, config.training_type, tokenizer,
                                                 device=self.device, seed=config.splitting_seed,
                                                 split=config.train_split_ratio)

            self.model.set_class_imbalance_weights(train_dataset.get_class_imbalance_weights(), device=self.device)

            val_dataset = EmotionCausalDataset(path, DatasetConfig.VAL, config.training_type, tokenizer,
                                               device=self.device, seed=config.splitting_seed,
                                               split=config.train_split_ratio)

            test_dataset = EmotionCausalDataset(path, DatasetConfig.TEST, config.training_type, tokenizer,
                                                device=self.device, seed=config.splitting_seed,
                                                split=config.train_split_ratio)

        elif config.solve_task == TaskSolve.TASK2:
            pass

        self.train_dataloader = self._get_dataloader(train_dataset, shuffle=True, for_training=True)
        self.val_dataloader = self._get_dataloader(val_dataset, shuffle=False, for_training=False)
        self.test_dataloader = self._get_dataloader(test_dataset, shuffle=False, for_training=False)

        self.writer = SummaryWriter(log_dir=os.path.join(config.base_path, config.model_log_path)) \
            if self.rank == 0 else None

        self.model = self._map_model_to_device()
        self._load_model_state()

    def _load_model_state(self, index=-1):
        """
        Load the model state from checkpoints if any.
        param index: Loads the checkpoint based on created date index. Defaults to -1 to load the latest checkpoint.
        """
        save_path = os.path.join(self.config.base_path, self.config.model_save_path)
        items = [os.path.join(save_path, item) for item in os.listdir(save_path) if '.pt' in item]

        total_steps = self.config.epochs * len(self.train_dataloader)
        # https://stackoverflow.com/a/61558319
        self.optim = AdamW(self.model.parameters(), lr=self.config.lr)
        warmup_steps = int(0.01 * len(self.train_dataloader))
        self.optim_lr_scheduler = get_constant_schedule_with_warmup(self.optim, num_warmup_steps=warmup_steps)

        if items:
            checkpoint_path = sorted(items, key=os.path.getctime)[index]

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if self.n_gpus > 1:
                self.model.module.load_state_dict(checkpoint['model'], strict=True)
            else:
                self.model.load_state_dict(checkpoint['model'], strict=True)

            self.model.train()
            self.optim.load_state_dict(checkpoint['optim'])
            self.optim_lr_scheduler.load_state_dict(checkpoint['optim_lr_scheduler'])

            self.config.current_epoch = 1 + checkpoint['epoch']

    def _save_model_state(self, current_epoch: int, **kwargs):
        path = os.path.join(self.config.base_path, self.config.model_save_path, f'{current_epoch}.pt')
        model = self.model.module if self.n_gpus > 1 else self.model

        torch.save({
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_lr_scheduler': self.optim_lr_scheduler.state_dict(),
            **kwargs
        }, f=path)

    def _map_model_to_device(self):
        """
        Maps model to correct device.
        """
        if self.n_gpus == 1:
            return self.model.to(self.device)
        elif self.n_gpus > 1:
            return DDP(self.model.to(self.rank), device_ids=[self.rank], output_device=self.rank,
                       find_unused_parameters=False)
        else:
            return self.model.to(self._cpu_device)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool, for_training: bool):
        sampler = None

        if self.config.multi_gpu and for_training:
            sampler = DistributedSampler(dataset, num_replicas=self.n_gpus, rank=self.rank,
                                         seed=self.config.splitting_seed, shuffle=shuffle)
            # Sampler cannot be applied with Shuffle with True.
            shuffle = False

        batch_size = self.config.train_batch_size if for_training else self.config.val_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

        return dataloader

    @staticmethod
    def _add_whitespace_after_punctuations(txt):
        n = len(txt)
        punctuations = [
            ',',
            '!',
            '?',
            '.',
            ';',
            '$',
            '&',
            '"',
            '...'
        ]

        if n < 3:
            return txt

        if txt[-1] in punctuations:
            txt = f'{txt[:-1]} {txt[-1]}'
        if txt[0] in punctuations:
            txt = f'{txt[0]} {txt[1:]}'

        inner = txt[1: -1]

        for char in punctuations:
            inner = inner.replace(char, f' {char} ')

        txt = f'{txt[0]}{inner}{txt[-1]}'
        # removing extra whitespaces.
        txt = txt.replace('  ', ' ')

        return txt

    def _extract_spans(self, span_logits: torch.Tensor, text_inp: dict) -> list:
        start_span_logit, end_span_logit = (x.squeeze(-1).contiguous() for x in span_logits.split(1, dim=-1))
        start_span_idx = start_span_logit.argmax(-1).tolist()
        end_span_idx = end_span_logit.argmax(-1).tolist()

        text = text_inp['input_ids']

        mask = torch.zeros_like(text)
        for batch_idx, (start_idx, end_idx) in enumerate(zip(start_span_idx, end_span_idx)):
            mask[batch_idx, start_idx: end_idx + 1] = 1.0

        masked_text_span = text * mask

        tokenizer = self.model.module.tokenizer() if self.n_gpus > 1 else self.model.tokenizer()
        pred_spans = tokenizer.batch_decode(masked_text_span, skip_special_tokens=True)
        pred_spans = [span.replace(" ##", "").replace("##", "").strip() for span in pred_spans]
        pred_spans = [self._add_whitespace_after_punctuations(span) for span in pred_spans]

        return pred_spans

    def _extract_spans_multilabel(self, span_logits: torch.Tensor, text_inp: dict, confidence: float) -> list:
        mask = span_logits >= confidence
        text = text_inp['input_ids']
        masked_text_span = text * mask

        tokenizer = self.model.module.tokenizer() if self.n_gpus > 1 else self.model.tokenizer()
        pred_spans = tokenizer.batch_decode(masked_text_span, skip_special_tokens=True)
        pred_spans = [span.replace(" ##", "").replace("##", "").strip() for span in pred_spans]
        pred_spans = [self._add_whitespace_after_punctuations(span) for span in pred_spans]

        return pred_spans

    def _extract_emotion_logits(self, emotion_logits: torch.Tensor) -> list:
        rev_emotion_label_map = self.val_dataloader.dataset.rev_emotion_labels
        emotion_idx = emotion_logits.argmax(-1)

        pred_emotion_labels = [rev_emotion_label_map[idx.item()] for idx in emotion_idx]

        return pred_emotion_labels

    @staticmethod
    def _accumulate_results(og_data: list, pred_emotion_labels: list, pred_spans: list) -> list:
        results = []

        for idx, data in enumerate(og_data):
            pred_emotion_label = pred_emotion_labels[idx]
            pred_span = pred_spans[idx] if pred_spans else ''

            results.append({
                'conversation_ID': data['conversation_id'],
                'utterance_ID': data['utterance_id_i'],
                'gold_emotion': data['emotion'],
                'predicted_emotion': pred_emotion_label,
                'compared_utterance_ID': data.get('utterance_id_j', -1),  # Not in use anymore.
                'predicted_text': pred_span,
                'gold_text': data.get('causal_span', '')
            })

        return results

    def train_task1(self):
        running_loss, running_acc = 0.0, 0.0
        disable_tqdm = self.rank != 0

        for epoch in range(self.config.current_epoch, self.config.epochs + 1):

            if self.train_dataloader.sampler:
                self.train_dataloader.sampler.set_epoch(epoch - 1)

            self.model.train()

            n_train = len(self.train_dataloader)

            global_step = (epoch - 1) * n_train
            with tqdm(total=n_train, colour='cyan', leave=True, disable=disable_tqdm) as bar:
                for idx, (inp, labels, _) in enumerate(self.train_dataloader, start=1):
                    self.optim.zero_grad()

                    if self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                        #                         s = inp['input_ids'].shape
                        #                         s1 = inp['attention_mask'].shape
                        #                         print(f'the labels are {s} {s1}')
                        #                         print(labels['label'][0])
                        out = self.model({'input_ids': inp['input_ids'], 'attention_mask': inp['attention_mask'],
                                          'labels': labels['label']})
                    else:
                        out = self.model(inp, labels)
                    loss = out['loss']
                    loss.backward()

                    self.optim.step()

                    if self.rank == 0:
                        running_loss += loss.item()
                        avg_loss = running_loss / idx
                        bar_string = f'Training {epoch}/{self.config.epochs} - Loss {avg_loss:.3f}'
                        self.writer.add_scalar('Loss/train', avg_loss, global_step=global_step)

                        if self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                            emotion_logits = out['emotion_logits']
                            pred_emotion_labels = emotion_logits
                            true_emotion_label = labels['label']
                            true_emotion_label = nn.functional.one_hot(true_emotion_label,
                                                                       num_classes=self.config.no_classes)
                            matched_emotions = (pred_emotion_labels == true_emotion_label)
                            running_acc += (matched_emotions.sum() / len(matched_emotions)).item()
                            bar_string = f'{bar_string} - emo_acc: {running_acc:.3f}'
                            self.writer.add_scalar('Emo_acc/train', running_acc, global_step=global_step)

                        bar.update()
                        bar.set_description(bar_string)

                        global_step += 1

            if self.rank == 0:
                self.writer.flush()

            self.optim_lr_scheduler.step()

            if self.n_gpus > 1:
                # Sync processes.
                dist.barrier()

                if self.rank == 0:
                    self._save_model_state(self.config.current_epoch)
            else:
                self._save_model_state(self.config.current_epoch)

            # Validation per epoch.
            if self.rank == 0:
                # reset _metrics for validation.
                self.model.eval()
                running_loss, running_acc = 0.0, 0.0
                thresholds = self.config.confidence_threshold
                results = [[] for _ in
                           range(len(thresholds))] if self.config.training_type == TrainingType.JOINT_TRAINING else []

                if self.n_gpus > 0:
                    torch.cuda.empty_cache()

                n_val = len(self.val_dataloader)

                global_step = (epoch - 1) * n_val
                processed_data = self.val_dataloader.dataset.processed_data
                og_data = self.val_dataloader.dataset.data

                model = self.model.module if self.n_gpus > 1 else self.model

                with tqdm(total=n_val, colour='red', leave=True) as bar:
                    for idx, (inp, labels, dataset_idx) in enumerate(self.val_dataloader, start=1):
                        with torch.no_grad():
                            if self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                                out = self.model(
                                    {'input_ids': inp['input_ids'], 'attention_mask': inp['attention_mask'],
                                     'labels': labels['label']})
                            else:
                                out = model(inp, labels)
                            loss = out['loss']

                            running_loss += loss.item()
                            avg_loss = running_loss / idx

                            pred_spans = []
                            pred_emotion_labels = []
                            processed_data_batch = [processed_data[idx] for idx in dataset_idx]

                            if self.config.training_type == TrainingType.JOINT_TRAINING:
                                emotion_logits = out['emotion_logits']
                                span_logits = out['span_logits']

                                # pred_spans = self._extract_spans(span_logits=span_logits, text_inp=inp)
                                pred_emotion_labels = self._extract_emotion_logits(emotion_logits=emotion_logits)
                                for idx, confidence in enumerate(thresholds):
                                    _pred_spans = self._extract_spans_multilabel(span_logits=span_logits,
                                                                                 text_inp=inp, confidence=confidence)
                                    pred_spans.append(_pred_spans)

                            elif self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                                emotion_logits = out['emotion_logits']
                                pred_emotion_labels = emotion_logits
                                true_emotion_label = labels['label']
                                true_emotion_label = nn.functional.one_hot(true_emotion_label,
                                                                           num_classes=self.config.no_classes)
                                matched_emotions = (pred_emotion_labels == true_emotion_label)
                                running_acc += (matched_emotions.sum() / len(matched_emotions)).item()


                            else:
                                span_logits = out['span_logits']
                                pred_spans = self._extract_spans(span_logits=span_logits, text_inp=inp)

                            if self.config.training_type == TrainingType.JOINT_TRAINING:

                                bar_str = f'Val {epoch}/{self.config.epochs} - Loss {avg_loss:.3f}'

                                for idx, (_pred_spans, confidence) in enumerate(zip(pred_spans, thresholds)):
                                    results[idx].extend(self._accumulate_results(processed_data_batch,
                                                                                 pred_emotion_labels, _pred_spans))
                                    _, metrics = evaluate_runtime(results[idx], og_data)
                                    weighted_prop_f1, micro_f1 = metrics[2], metrics[-1]

                                    self.writer.add_scalar(f'W_prop_F1_conf_{confidence}/val',
                                                           weighted_prop_f1, global_step)
                                    self.writer.add_scalar(f'micro_F1_conf_{confidence}/val',
                                                           micro_f1, global_step)
                                    # Show the middle one
                                    if idx == (len(thresholds) // 2):
                                        bar_str = f'{bar_str} W. prop. F1 @ conf {confidence} - {weighted_prop_f1:.3f}'
                            else:
                                results.extend(self._accumulate_results(processed_data_batch,
                                                                        pred_emotion_labels, pred_spans))
                                _, metrics = evaluate_runtime(results, og_data)
                                weighted_prop_f1, micro_f1 = metrics[2], metrics[-1]

                                self.writer.add_scalar('W_prop_F1/val', weighted_prop_f1, global_step)
                                self.writer.add_scalar('micro_F1/val', micro_f1, global_step)

                                bar_str = (f'Val {epoch}/{self.config.epochs} - Loss '
                                           f'{avg_loss:.3f} W. prop. F1 {weighted_prop_f1:.3f}')

                            self.writer.add_scalar('Loss/val', avg_loss, global_step)

                            bar.update()
                            bar.set_description(bar_str)
                            global_step += 1

                if self.config.training_type == TrainingType.JOINT_TRAINING:
                    results = dict(zip(thresholds, results))

                with open(f"validation_{epoch}.json", "w") as f:
                    validation_logs = {f"results_{epoch}": results}
                    json.dump(validation_logs, f, indent=4)

                self.writer.flush()

            if self.n_gpus > 0:
                torch.cuda.empty_cache()
                if self.n_gpus > 1:
                    dist.barrier()

            self.config.current_epoch += 1
            running_loss, running_acc = 0.0, 0.0

        if self.rank == 0:
            self.writer.close()

        if self.n_gpus > 1:
            dist.barrier()
            cleanup()

    @torch.no_grad()
    def evaluate_task1(self, threshold: float = None):
        if self.config.training_type != TrainingType.JOINT_TRAINING:
            raise NotImplementedError(f'Not implemented testing for other training types')

        if threshold is None:
            n = len(self.config.confidence_threshold)
            threshold = self.config.confidence_threshold[n // 2]

        self.model.eval()
        model = self.model.module if self.n_gpus > 1 else self.model
        n_test = len(self.test_dataloader)
        processed_data = self.test_dataloader.dataset.processed_data

        running_loss = 0.0
        results = []

        with tqdm(total=n_test, colour='red', leave=True) as bar:
            for idx, (inp, dataset_idx) in enumerate(self.test_dataloader, start=1):
                out = model(inp, {})
                loss = out['loss']
                emotion_logits = out['emotion_logits']
                span_logits = out['span_logits']

                running_loss += loss.item()
                avg_loss = running_loss / idx
                processed_data_batch = [processed_data[d_idx] for d_idx in dataset_idx]
                pred_emotion_labels = self._extract_emotion_logits(emotion_logits=emotion_logits)
                bar = f'Test Loss {avg_loss:.3f}'
                pred_spans = self._extract_spans_multilabel(span_logits=span_logits, text_inp=inp, confidence=threshold)

                results.extend([
                    {
                        'conversation_ID': processed_data_batch[b_idx]['conversation_id'],
                        'utterance_ID': processed_data_batch[b_idx]['utterance_id_i'],
                        'text': pred_spans[b_idx],
                        'emotion': pred_emotion_labels
                    }
                    for b_idx in range(self.config.val_batch_size)
                ])


def multi_gpu_train(rank, config: TrainerConfig, n_gpus: int):
    trainer = Trainer(config, rank=rank, n_gpus=n_gpus)
    trainer.train_task1()


def joint_model_config():
    return TrainerConfig()


def emotion_classification_config():
    return TrainerConfig(training_type=TrainingType.EMOTION_CLASSIFICATION)


def span_classification_config():
    return TrainerConfig(training_type=TrainingType.SPAN_CLASSIFICATION)


if __name__ == '__main__':
    trainer_config = joint_model_config()
    n_procs = torch.cuda.device_count()

    if trainer_config.multi_gpu and n_procs > 1:
        mp.spawn(multi_gpu_train, args=(trainer_config, n_procs), nprocs=n_procs)
    else:
        single_trainer = Trainer(trainer_config, n_gpus=n_procs)
        single_trainer.train_task1()