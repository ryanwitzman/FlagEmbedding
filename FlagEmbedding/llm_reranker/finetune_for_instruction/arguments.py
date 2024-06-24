import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer import *
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

def default_list() -> List[str]:
    return ["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_model_path: str = field(default='')
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of lora."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of lora modules."})
    target_modules: List[str] = field(default_factory=default_list)
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
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
    cache_dir: str = field(default="tmp", metadata={"help": "the cache of the model"})
    token: str = field(default=None, metadata={"help": "the token to access the huggingface model"})
    from_peft: str = field(default=None)
    lora_extra_parameters: str = field(default=None)
    eval_model_path: str = field(
        default=None,
        metadata={"help": "Path to the model to be evaluated, if different from the training model"}
    )

@dataclass
class DataArguments:
    train_data: str = field(
        default='toy_finetune_data.jsonl', metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)
    query_max_len: int = field(
        default=32,
        metadata={"help": "The maximum total input sequence length after tokenization for passage."}
    )
    passage_max_len: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization for passage."}
    )
    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )
    query_instruction_for_retrieval: str = field(default="A: ", metadata={"help": "query: "})
    passage_instruction_for_retrieval: str = field(default="B: ", metadata={"help": "passage: "})
    cache_path: str = field(default='./data_dir')
    load_from_disk: bool = field(default=False, metadata={"help": " whether load the data from disk"})
    load_disk_path: str = field(default=None, metadata={"help": " the path to load the data", "nargs": "+"})
    save_to_disk: bool = field(default=False, metadata={"help": " whether save the data to disk"})
    save_disk_path: str = field(default=None, metadata={"help": " the path to save the data"})
    num_shards: int = field(
        default=0,
        metadata={"help": "number of shards to write, prior than `save_max_shard_size`"}
    )
    save_max_shard_size: str = field(default="50GB", metadata={"help": "the max size of the shard"})
    exit_after_save: bool = field(default=False, metadata={"help": " whether exit after save the data"})
    eval_data: str = field(default=None, metadata={"help": "Path to evaluation data"})
    eval_group_size: int = field(
        default=8, metadata={"help": "Group size for evaluation, if different from training"}
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."}
    )
    eval_metric: str = field(
        default="accuracy",
        metadata={"help": "Metric to use for evaluation. Options: 'accuracy', 'f1', 'precision', 'recall'"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
        if self.eval_data and not os.path.exists(self.eval_data):
            raise FileNotFoundError(f"cannot find file: {self.eval_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    loss_type: str = field(default='only logits')
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    evaluate_during_training: bool = field(default=False, metadata={"help": "Run evaluation during training at each logging step."})

class BiTrainer(Trainer):
    use_lora: bool

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if not self.use_lora:
            super()._save(output_dir, state_dict)
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'model.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Save adapter model at {output_dir}")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Initialize containers for evaluation results
        all_losses = []
        all_preds = []
        all_labels = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        for batch in eval_dataloader:
            with torch.no_grad():
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                loss = outputs.loss
                all_losses.append(loss.item())
                
                # Get predictions and labels
                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute average loss
        avg_loss = sum(all_losses) / len(all_losses)
        
        # Compute accuracy
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        
        # Create metrics dict
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
        }
        
        # Log results
        self.log(metrics)
        
        return metrics
