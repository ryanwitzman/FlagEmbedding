from transformers.trainer import Trainer, EvalLoopOutput, EvalPrediction
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
import torch
import os
import logging
from typing import Optional,List
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
            batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1
            num_examples = self.num_examples(dataloader)
            logger.info(f"***** Running {description} *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Batch size = {batch_size}")
            model.eval()
            self.callback_handler.eval_dataloader = dataloader
            
            total_loss = 0.0
            num_batches = 0
            
            for step, inputs in enumerate(dataloader):
                with torch.no_grad():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    if isinstance(loss, dict):
                        # If loss is a dictionary, extract the scalar loss value
                        if 'loss' in loss:
                            loss_value = loss['loss']
                        elif 'scores' in loss:
                            # Assuming 'scores' might be used as a proxy for loss
                            loss_value = loss['scores'].mean() if torch.is_tensor(loss['scores']) else np.mean(loss['scores'])
                        else:
                            logger.warning(f"Unexpected loss dictionary format: {loss.keys()}")
                            continue
                    else:
                        loss_value = loss
                    
                    if torch.is_tensor(loss_value):
                        loss_value = loss_value.item()
                    
                    total_loss += loss_value
                    num_batches += 1
    
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            metrics = {f"{metric_key_prefix}_loss": avg_loss}
            self.log(metrics)
            
            return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_examples)
