from transformers.trainer import *
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union

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
        
        loss_host = torch.tensor(0.0).to(self.args.device)
        all_losses = torch.tensor(0.0).to(self.args.device)
        all_preds = None
        all_labels = None
        
        observed_num_examples = 0
        
        for step, inputs in enumerate(dataloader):
            # Handle different input types
            if isinstance(inputs, dict):
                batch = [inputs]
            elif isinstance(inputs, list) and all(isinstance(item, dict) for item in inputs):
                batch = inputs
            else:
                raise ValueError(f"Unexpected input type: {type(inputs)}. Expected dict or list of dicts.")

            observed_batch_size = len(batch)
            observed_num_examples += observed_batch_size
            
            with torch.no_grad():
                batch_losses = []
                batch_preds = []
                batch_labels = []
                for instance in batch:
                    instance = self._prepare_inputs(instance)
                    loss, outputs = self.compute_loss(model, instance, return_outputs=True)
                    batch_losses.append(loss.detach())
                    batch_preds.append(outputs.logits.detach().cpu().numpy())
                    batch_labels.append(instance["labels"].detach().cpu().numpy())
                
                loss_host += sum(batch_losses)
                all_losses += sum(batch_losses)
            
            if all_preds is None:
                all_preds = np.concatenate(batch_preds, axis=0)
                all_labels = np.concatenate(batch_labels, axis=0)
            else:
                all_preds = np.concatenate([all_preds] + batch_preds, axis=0)
                all_labels = np.concatenate([all_labels] + batch_labels, axis=0)
            
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
        
        loss = loss_host / len(dataloader)
        all_losses = all_losses / observed_num_examples
        
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics[f"{metric_key_prefix}_loss"] = all_losses.item()
        
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log(metrics)
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_examples)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        self.log(output.metrics)
        
        if self.args.tpu_metrics_debug or self.args.debug:
            if is_torch_tpu_available():
                xm.master_print(met.metrics_report())
            elif is_sagemaker_mp_enabled():
                smp.push_metrics_to_sagemaker()
        
        return output.metrics

def find_batch_size(inputs):
    for v in inputs.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    return None
