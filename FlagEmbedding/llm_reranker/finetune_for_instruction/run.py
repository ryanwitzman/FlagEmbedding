import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForReranker, RerankCollator, EvalDatasetForReranker
from .modeling import BiEncoderModel
from .trainer import BiTrainer
from .load_model import get_model

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = 1
        base_model = get_model(model_args, training_args)
        logger.info("Base model loaded successfully")

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
            trust_remote_code=True,
            token=model_args.token,
            add_eos_token=True
        )
        logger.info("Tokenizer loaded successfully")

        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'mistral' in model_args.model_name_or_path.lower():
            tokenizer.padding_side = 'left'

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
        logger.info('Config: %s', config)

        model = BiEncoderModel(model=base_model,
                               tokenizer=tokenizer,
                               train_batch_size=training_args.per_device_train_batch_size)
        logger.info("BiEncoderModel created successfully")

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        train_dataset = TrainDatasetForReranker(args=data_args, tokenizer=tokenizer)
        logger.info(f"Train dataset loaded. Size: {len(train_dataset)}")

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = EvalDatasetForReranker(args=data_args, tokenizer=tokenizer)
            logger.info(f"Eval dataset loaded. Size: {len(eval_dataset)}")

        data_collator = RerankCollator(
            tokenizer=tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )

        trainer = BiTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.use_lora = model_args.use_lora

        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        if training_args.do_train:
            logger.info("*** Start Training ***")
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            logger.info("*** Training completed ***")
            
            # Log and save training results
            logger.info(f"*** Training metrics: {train_result.metrics} ***")
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Start Evaluation ***")
            metrics = trainer.evaluate()
            logger.info(f"*** Evaluation metrics: {metrics} ***")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if not model_args.use_lora:
            checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
            trainer.deepspeed.save_checkpoint(checkpoint_dir)
            logger.info(f"Final checkpoint saved to {checkpoint_dir}")

        # Save tokenizer
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info(f"Tokenizer saved to {training_args.output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
