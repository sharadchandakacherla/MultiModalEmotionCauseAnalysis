import os
from typing import Union, Type

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import TrainerConfig, TrainingType, TaskSolve
from dataset import DatasetConfig, EmotionCausalDataset
from models import ModelBaseClass, JointModel, EmotionClassification, SpanClassification


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


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.model_log_path, exist_ok=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_cls = select_model(config.training_type, config.solve_task)

        self.model = model_cls(base_model_name=config.base_model_name, no_classes=config.base_model_name) \
            .to(self.device)

        if config.solve_task == TaskSolve.TASK1:
            self.model.add_special_token_to_tokenizer(config.special_token)

        if config.freeze_base_model:
            self.model.freeze_base_model()

        if config.solve_task == TaskSolve.TASK1:
            path = os.path.join(config.base_path, 'data', 'text')
            tokenizer = self.model.tokenizer()
            train_dataset = EmotionCausalDataset(path, DatasetConfig.TRAIN, config.training_type, tokenizer,
                                                 device=self.device, seed=config.splitting_seed,
                                                 split=config.train_split_ratio)

            val_dataset = EmotionCausalDataset(path, DatasetConfig.VAL, config.training_type, tokenizer,
                                               device=self.device, seed=config.splitting_seed,
                                               split=config.train_split_ratio)

            test_dataset = EmotionCausalDataset(path, DatasetConfig.TEST, config.training_type, tokenizer,
                                                device=self.device, seed=config.splitting_seed,
                                                split=config.train_split_ratio)
        elif config.solve_task == TaskSolve.TASK2:
            pass

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.val_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.val_batch_size, shuffle=False)

        self.writer = SummaryWriter(log_dir=config.model_log_path)

        total_steps = config.epochs * len(self.train_dataloader)

        self.optim = AdamW(self.model.parameters())
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
        save_path = self.config.model_save_path
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
        path = os.path.join(self.config.model_save_path, f'{current_epoch}.pt')
        torch.save({
            'epoch': current_epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_lr_scheduler': self.optim_lr_scheduler.state_dict(),
            **kwargs
        }, f=path)

    def train_task1(self):
        running_loss, running_acc = 0.0, 0.0

        for epoch in range(self.config.current_epoch, self.config.epochs + 1):
            self.model.train()

            global_step = (epoch - 1) * len(self.train_dataloader)
            with tqdm(total=len(self.train_dataloader), colour='cyan', leave=False) as bar:
                for idx, batch in enumerate(self.train_dataloader, start=1):
                    self.optim.zero_grad()

                    out = self.model(**batch)
                    loss = out['loss']
                    loss.backward()

                    self.optim.step()

                    running_loss += loss.item()
                    avg_loss = running_loss / idx
                    avg_emo_acc = None

                    # TODO: Evaluate metrics.
                    if self.config.training_type == TrainingType.JOINT_TRAINING:
                        emotion_logits = out['emotion_logits']
                        # Calculate emotion acc
                        # running_acc = ...
                        avg_emo_acc = running_acc / idx
                        span_logits = out['span_logits']
                        # Calculate span based metric?
                    elif self.config.training_type == TrainingType.EMOTION_CLASSIFICATION:
                        emotion_logits = out['emotion_logits']
                        # Calculate emotion acc
                        # runnning_acc = ...
                        avg_emo_acc = running_acc / idx
                    else:
                        span_logits = out['span_logits']
                        # Calculate span based metric?

                    bar_string = f'Training {epoch}/{self.config.epochs} - Loss {avg_loss:.3f}'
                    self.writer.add_scalar('Loss/train', avg_loss, global_step=global_step)
                    if avg_emo_acc is not None:
                        bar_string = f'{bar_string} - emo_acc: {avg_emo_acc:.3f}'
                        self.writer.add_scalar('Loss/emo_acc', avg_emo_acc, global_step=global_step)

                    global_step += 1

                    bar.update()
                    bar.set_description(bar_string)

            self.optim_lr_scheduler.step()

    def evaluate_task1(self):
        pass
