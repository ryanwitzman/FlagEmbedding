from transformers.trainer import *
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BiTrainer(Trainer):
    use_lora: bool
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
    self,
    dataloader: DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running evaluation *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        
        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**inputs)
                # Ensure that the loss is extracted correctly
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", None)
                    scores = outputs.get("scores", None)
                else:
                    loss = outputs.loss
                    scores = outputs.scores
                if loss is not None:
                    total_loss += loss.item()
                
                if not prediction_loss_only and scores is not None:
                    all_preds.append(scores.cpu().numpy())
                    if 'labels' in inputs:
                        all_labels.append(inputs['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = {}
        metrics[f"{metric_key_prefix}_loss"] = avg_loss
        
        if not prediction_loss_only:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            if self.compute_metrics is not None:
                metrics.update(self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels)))
        
        self.log(metrics)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_examples)
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        return self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        ).metrics
def find_batch_size(inputs):
    for v in inputs.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    return None
