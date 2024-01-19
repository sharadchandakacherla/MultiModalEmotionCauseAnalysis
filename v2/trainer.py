import os
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import JointModel, EmotionClassifcation, SpanClassification
from config import TrainerConfig, TrainingType, TaskSolve
from dataset import DatasetConfig, EmotionCausalDatatset


def select_model(training_type: TrainingType):
    if training_type == TrainingType.JOINT_TRAINING:
            model = JointModel
    elif training_type == TrainingType.EMOTION_CLASSIFICATION:
        model = EmotionClassifcation
    elif training_type == TrainingType.SPAN_CLASSIFICATION:
        model = SpanClassification
    else:
        raise NotImplementedError(f'ill-defined training type {self.config.training_type.}')
        
    return model


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_cls = select_model(config.training_type)
            
        self.model = model_cls(base_model_name=config.base_model_name, no_classes=config.base_model_name)
        
        if config.freeze_base_model:
            self.model.freeze_base_model()
            
        self.optim = AdamW(self.model.parameters())
        # https://stackoverflow.com/a/61558319
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=0, num_training_steps=None)