from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import TrainingArguments, Trainer
import random
import torch
import argparse
import numpy as np


def format_dataset(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["contexts"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["answers"] = {}
    example["answers"]['text'] = example["answer"]["value"]
    example["answers"]["start_char"] = example["contexts"].find(example["answers"]['text'])
    example["answers"]["end_char"] = example["answers"]["start_char"] + len(example["answers"]['text'])
    return example

def preprocess_training_examples(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["contexts"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    #For each sub-token returned by the tokenizer, the offset mapping gives us a 
    # tuple indicating the sub-token's start position and end position relative to the original token it was split from.
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['start_char']
        end_char = answer['end_char']
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["contexts"],
        max_length=384,
        truncation=True,
        stride=50,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["question_id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def process_dataset(tokenizer, perc_dataset):
	trivia = load_dataset("trivia_qa","rc.wikipedia") 

	trivia_small = trivia['train'].select(range(int(perc_dataset / 100 * len(trivia['train']))))
	eval_small = trivia['validation'].select(range(int(perc_dataset / 100 * len(trivia['validation']))))

	prova_small =  trivia_small.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer"])
	prova_eval_small =  eval_small.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer"])

	train_dataset = prova_small.map(
	        (lambda examples: preprocess_training_examples(examples, tokenizer)),
	        batched=True,
	        remove_columns=prova_small.column_names,
	    )

	validation_dataset = prova_eval_small.map(
	    (lambda examples: preprocess_validation_examples(examples, tokenizer)),
	    batched=True,
	    remove_columns=prova_eval_small.column_names,
	    )

	return train_dataset, validation_dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', default=0,
                        help='Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--trivia_perc', default=5.0, type=float, help='The percentage of dataset to consider')
    parser.add_argument('--freq_eval', default=500, type=int, help="frequency w.r.t. global_step to perform evaluation")
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--checkpoints_dir', default=None, type=str)
    parser.add_argument('--resume_training', default=False, type=bool)

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

	train_dataset, validation_dataset = process_dataset(tokenizer, args.trivia_perc)

	training_args = TrainingArguments(
	    output_dir=args.checkpoints_dir,    									# output directory
	    num_train_epochs=args.num_train_epochs,             					# total number of training epochs
	    per_device_train_batch_size=args.train_batch_size,  					# batch size per device during training
	    per_device_eval_batch_size=args.eval_batch_size,   						# batch size for evaluation
	    warmup_steps=args.warmup_steps,                 						# number of warmup steps for learning rate scheduler
	    weight_decay=args.weight_decay,               							# strength of weight decay
	    logging_dir=args.log_dir,            									# directory for storing logs
	    logging_steps=10,
	    evaluation_strategy="no",
	    save_strategy="steps",
	    save_steps=args.freq_eval,
	    save_total_limit=2,
	)

	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=validation_dataset,
	    tokenizer=tokenizer,
	)
	trainer.train(args.resume_training)

	trainer.save_model(f"{args.checkpoints_dir}/training_trivia_finished")

	print("Training done!")