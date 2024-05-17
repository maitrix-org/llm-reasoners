from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from transformers import TrainingArguments

TASKS = ["gsm8k", "mathqa", "multiarith", "svamp", "last_letter_concatenation", "coin_flip", "tso"]

@dataclass
class DiscriminatorModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    pooling: Optional[str] = field(
        default="max",
        metadata={
            "help": (
                "The pooling strategy for the encoder outputs. Can be 'mean' or 'max'."
            )
        },
    )


@dataclass
class DiscriminatorDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    model_style : Optional[str] = field(
        default="enc",
        metadata={       
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        
        },
    )
    task: Optional[str] = field(
        default=None, metadata={"help": "The name of the task to train on: " + ", ".join(TASKS),
                                "choices": TASKS 
                            }
    )
    trajectory_path: Optional[str] = field(default=None, metadata={"help": "Path to the model trajectories file"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_len: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    n_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training and eval examples to this "
                "value if set."
            )
        },
    )
    dev_is_train: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the dev set is used as the train set. Useful for debugging purposes."
            )
        },
    )

    invalid_prefix_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "The probability of replacing a valid prefix with an invalid one."
            )
        },
    )

    max_alignment_cost: Optional[float] = field(
        default=1.8,
        metadata={
            "help": (
                "The probability of replacing a valid prefix with an invalid one."
            )
        },
    )

    step_aligner_model: Optional[str] = field(
        default="roscoe",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    step_delimiter: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The delimiter to use for the step alignment. If not provided the default for each task will be used."
            )
        },
    )

    break_after_extra_step: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the alignment will stop after the first extra step."
            )
        },
    )

    use_correct_samples: Optional[bool] = field(
        default=False,
        metadata={
            "help": ( 
                "If true, the correct samples are used for the alignment." 
            )
        },
    )
    skip_alignment: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the alignment is skipped."
            )
        },
    )


@dataclass
class DiscriminatorTrainingArguments(TrainingArguments):

    ## add arguments for the discriminator
    margin: Optional[float] = field(
        default=0.1,
        metadata={        
            "help": (
                "The maximum margin for the discriminator loss"
            )
        },
    )

    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={        
            "help": (
                "The checkpoint to load"
            )
        },
    )

    seed: Optional[int] = field(
        default=7,
        metadata={        
            "help": (
                "The seed to use"
            )
        },
    )
    
    loss_type: Optional[str] = field(
        default="maxmargin",
        metadata={
            "help": (
                "The loss type to use for the discriminator. Can be 'maxmargin' or 'logsigmoid'."
            ),
            "choices": ["maxmargin", "logsigmoid"]
        },
    )

    fix_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the tokenizer is not added new tokens. Used with few-shot setting." 
            )
        },
    )

    dev_metric: Optional[str] = field(
        default="loss",
        metadata={
            "help": (
                "The metric to use for early stopping."
            ),
            "choices": ["acc", "loss"]
        },
    )


