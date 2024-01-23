import os
from dataclasses import dataclass
from enum import IntEnum


class TrainingType(IntEnum):
    JOINT_TRAINING = 1
    EMOTION_CLASSIFICATION = 2
    SPAN_CLASSIFICATION = 3


class TaskSolve(IntEnum):
    TASK1 = 1
    TASK2 = 2


@dataclass
class TrainerConfig:
    base_path: str = os.getcwd()
    model_log_path: str = 'roberta-base_log_exp1'
    model_save_path: str = 'roberta-base_save_exp1'
    base_model_name: str = 'mrm8488/spanbert-finetuned-squadv2'
    no_classes: int = 7 # 6 + 1
    train_split_ratio: float = 0.8
    splitting_seed: int = 42
    epochs: int = 10
    current_epoch: int = 1
    lr: float = 1e-3
    train_batch_size: int = 4 * 8
    val_batch_size: int = 4 * 8
#     is_train: bool = True
    training_type: TrainingType = TrainingType.JOINT_TRAINING
    freeze_base_model: bool = False
    solve_task: TaskSolve = TaskSolve.TASK1
    special_token: str = '<SEP>'
    multi_gpu: bool = True
    using_confidence_threshold: bool = True
    confidence_threshold: tuple = (0.5, 0.75, 0.9)
