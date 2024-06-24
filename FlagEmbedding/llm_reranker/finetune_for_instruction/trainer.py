from transformers.trainer import Trainer, EvalLoopOutput, EvalPrediction
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
import torch
import os
import logging
from typing import Optional
import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

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
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader

        all_scores = []
        all_labels = []

        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.get("scores", None)
                if scores is not None:
                    all_scores.append(scores.cpu().numpy())
                    if 'labels' in inputs:
                        all_labels.append(inputs['labels'].cpu().numpy())

        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

        metrics = {}
        if self.compute_metrics is not None:
            metrics.update(self.compute_metrics(EvalPrediction(predictions=all_scores, label_ids=all_labels)))

        self.log(metrics)
        return EvalLoopOutput(predictions=all_scores, label_ids=all_labels, metrics=metrics, num_samples=num_examples)
