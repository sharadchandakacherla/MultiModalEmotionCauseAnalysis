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
    model_log_path: str = ''
    model_save_path: str = ''
    base_model_name: str = ''
    no_classes: int = 7 # 6 + 1
    train_split_ratio: float = 0.8
    splitting_seed: int = 42
    epochs: int = 10
    current_epoch: int = 1
    lr: float = 1e-3
    train_batch_size: int = 4
    val_batch_size: int = 4
#     is_train: bool = True
    training_type: TrainingType = TrainingType.JOINT_TRAINING
    freeze_base_model: bool = False
    solve_task: TaskSolve = TaskSolve.TASK1
    special_token: str = '<SEP>'

