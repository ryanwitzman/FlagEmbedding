import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="BAAI/bge-reranker-v2-gemma",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "It is an option to create the model as an empty shell, "
                          "then only materialize its parameters when the pretrained weights are loaded."}
    )
    cache_dir: str = field(
        default="tmp", metadata={"help": "the cache of the model"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default='train.json', metadata={"help": "Path to train data"}
    )
    eval_data: str = field(
        default='eval.json', metadata={"help": "Path to evaluation data"}
    )
    train_group_size: int = field(default=8)
    eval_group_size: int = field(default=8)
    query_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query."
        },
    )
    passage_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    query_instruction_for_retrieval: str = field(
        default="A: ", metadata={"help": "query instruction"}
    )
    passage_instruction_for_retrieval: str = field(
        default="B: ", metadata={"help": "passage instruction"}
    )


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    output_dir: str = field(default="gemma-finetune-2")
    report_to: str = field(default="wandb")
    num_train_epochs: float = field(default=2.0)
    learning_rate: float = field(default=2e-4)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    warmup_ratio: float = field(default=0.0)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=50)
    save_steps: int = field(default=50)
    max_grad_norm: float = field(default=1.0)
    dataloader_drop_last: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    overwrite_output_dir: bool = field(default=True)
    loss_type: str = field(default="only logits")
    evaluate_during_training: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        self.ddp_find_unused_parameters = False  # This is set explicitly in the command

if __name__ == "__main__":
    # This part is just to demonstrate how to use these dataclasses
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = RetrieverTrainingArguments(output_dir="gemma-finetune-2")
    
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    print(f"Training arguments: {training_args}")
