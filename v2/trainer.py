import os
from typing import Union, Type

import torch
from torch import multiprocessing as mp
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import TrainerConfig, TrainingType, TaskSolve
from dataset import DatasetConfig, EmotionCausalDataset
from models import ModelBaseClass, JointModel, EmotionClassification, SpanClassification
from v2.metrics.evaluate import evaluate_runtime


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

        self.model = self._map_model_to_device()

        if config.solve_task == TaskSolve.TASK1:
            path = os.path.join(config.base_path, 'data', 'text')
            tokenizer = self.model.tokenizer()
            train_dataset = EmotionCausalDataset(path, DatasetConfig.TRAIN, config.training_type, tokenizer,
                                                 device=self.device, seed=config.splitting_seed,
                                                 split=config.train_split_ratio)

            val_dataset = EmotionCausalDataset(path, DatasetConfig.VAL, config.training_type, tokenizer,
                                               device=self.device, seed=config.splitting_seed,
                                               split=config.train_split_ratio)

        #             test_dataset = EmotionCausalDataset(path, DatasetConfig.TEST, config.training_type, tokenizer,
        #                                                 device=self.device, seed=config.splitting_seed,
        #                                                 split=config.train_split_ratio)
        elif config.solve_task == TaskSolve.TASK2:
            pass

        self.train_dataloader = self._get_dataloader(train_dataset, shuffle=True, for_training=True)
        self.val_dataloader = self._get_dataloader(val_dataset, shuffle=False, for_training=False)
        # self.test_dataloader = self._get_dataloader(test_dataset, shuffle=False, for_training=False)

        # self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)
        # self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.val_batch_size, shuffle=False)
        #         self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.val_batch_size, shuffle=False)

        self.writer = SummaryWriter(log_dir=os.path.join(config.base_path, config.model_log_path)) \
            if self.rank == 0 else None

        total_steps = config.epochs * len(self.train_dataloader)

        self.optim = AdamW(self.model.parameters(), lr=config.lr)
        # https://stackoverflow.com/a/61558319
        warmup_steps = int(0.1 * len(self.train_dataloader))
        self.optim_lr_scheduler = get_cosine_schedule_with_warmup(self.optim, num_warmup_steps=warmup_steps,
                                                                  num_training_steps=total_steps)

        self._load_model_state()

    def _load_model_state(self, index=-1):
        """
        Load the model state from checkpoints if any.
        param index: Loads the checkpoint based on created date index. Defaults to -1 to load the latest checkpoint.
        """
        save_path = os.path.join(self.config.base_path, self.config.model_save_path)
        items = [os.path.join(save_path, item) for item in os.listdir(save_path)]
        if items:
            checkpoint_path = sorted(items, key=os.path.getctime)[index]

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
            setup(self.rank, self.n_gpus)
            sampler = DistributedSampler(dataset, num_replicas=self.n_gpus, rank=self.rank,
                                         seed=self.config.splitting_seed, shuffle=shuffle)
            # Sampler cannot be applied with Shuffle with True.
            shuffle = False

        batch_size = self.config.train_batch_size if for_training else self.config.val_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

        return dataloader

    def _extract_spans(self, span_logits: torch.Tensor, text_inp: dict) -> list:
        start_span_logit, end_span_logit = (x.squeeze(-1).contiguous() for x in span_logits.split(1, dim=-1))
        start_span_idx = start_span_logit.argmax(-1)
        end_span_idx = end_span_logit.argmax(-1)

        text_span_mask = torch.zeros_like(text_inp['input_ids'])
        for batch_idx, (start_idx, end_idx) in enumerate(zip(start_span_idx, end_span_idx)):
            text_span_mask[batch_idx, start_idx: end_idx + 1] = 1.0

        pred_spans = self.model.tokenizer.batch_decode(text_span_mask, skip_special_tokens=True)

        return pred_spans

    def _extract_emotion_logits(self, emotion_logits: torch.Tensor) -> list:
        rev_emotion_label_map = self.val_dataloader.dataset.rev_emotion_labels
        emotion_idx = emotion_logits.argmax(-1)

        pred_emotion_labels = [rev_emotion_label_map[idx] for idx in emotion_idx]

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
                'compared_utterance_ID': data.get('utterance_id_j', -1),
                'predicted_text': pred_span,
                'gold_text':  data.get('causal_span', '')
            })

        return results

    def train_task1(self):
        running_loss, running_acc = 0.0, 0.0
        disable_tqdm = self.rank != 0

        for epoch in range(self.config.current_epoch, self.config.epochs + 1):

            if self.train_dataloader.sampler:
                self.train_dataloader.sampler.set_epoch(epoch - 1)

            self.model.train()

            global_step = (epoch - 1) * len(self.train_dataloader)
            with tqdm(total=len(self.train_dataloader), colour='cyan', leave=False, disable=disable_tqdm) as bar:
                for idx, (inp, labels, _) in enumerate(self.train_dataloader, start=1):
                    self.optim.zero_grad()

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
                            avg_emo_acc = running_acc / idx
                            bar_string = f'{bar_string} - emo_acc: {avg_emo_acc:.3f}'
                            self.writer.add_scalar('Loss/emo_acc', avg_emo_acc, global_step=global_step)

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
                results = []

                if self.n_gpus > 0:
                    torch.cuda.empty_cache()

                global_step = (epoch - 1) * len(self.val_dataloader)
                processed_data = self.val_dataloader.dataset.processed_data
                og_data = self.val_dataloader.dataset.data

                with tqdm(total=len(self.val_dataloader), colour='red', leave=False) as bar:
                    for idx, (inp, labels, dataset_idx) in enumerate(self.val_dataloader, start=1):
                        with torch.no_grad():
                            out = self.model(inp, labels)
                            loss = out['loss']

                            running_loss += loss.item()
                            avg_loss = running_loss / idx

                            pred_spans = []
                            pred_emotion_labels = []
                            processed_data_batch = [processed_data[idx] for idx in dataset_idx]

                            if self.config.training_type == TrainingType.JOINT_TRAINING:
                                emotion_logits = out['emotion_logits']
                                span_logits = out['span_logits']

                                pred_spans = self._extract_spans(span_logits=span_logits, text_inp=inp)
                                pred_emotion_labels = self._extract_emotion_logits(emotion_logits=emotion_logits)

                            elif self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                                emotion_logits = out['emotion_logits']
                                pred_emotion_labels = self._extract_emotion_logits(emotion_logits=emotion_logits)
                            else:
                                span_logits = out['span_logits']
                                pred_spans = self._extract_spans(span_logits=span_logits, text_inp=inp)

                            results.extend(self._accumulate_results(processed_data_batch,
                                                                    pred_emotion_labels, pred_spans))

                            _, metrics = evaluate_runtime(results, og_data)
                            weighted_prop_f1 = metrics[2]

                            bar.update()
                            bar.set_description(f'Val {epoch}/{self.config.epochs} - Loss {avg_loss:.3f} '
                                                f'W. prop. F1 {weighted_prop_f1:.3f}')

                            self.writer.add_scalar('Loss/val', avg_loss, global_step)
                            self.writer.add_scalar('W_prop_F1/val', weighted_prop_f1, global_step)
                            global_step += 1

                self.writer.flush()
                results.clear()

            if self.n_gpus > 0:
                torch.cuda.empty_cache()
                if self.n_gpus > 1:
                    dist.barrier()

            self.config.current_epoch += 1

        if self.rank == 0:
            self.writer.close()

        if self.n_gpus > 1:
            dist.barrier()
            cleanup()

    def evaluate_task1(self):
        pass


def multi_gpu_train(rank, config: TrainerConfig, n_gpus: int):
    trainer = Trainer(config, rank=rank, n_gpus=n_gpus)
    trainer.train_task1()


if __name__ == '__main__':
    trainer_config = TrainerConfig()
    n_procs = torch.cuda.device_count()

    if trainer_config.multi_gpu:
        mp.spawn(multi_gpu_train, args=(trainer_config,), nprocs=n_procs)
    else:
        single_trainer = Trainer(trainer_config, n_gpus=n_procs)
        single_trainer.train_task1()
