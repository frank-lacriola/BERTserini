from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer,TrainingArguments, Trainer
from datasets import load_dataset
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult
from tqdm.auto import tqdm
import collections
import numpy as np 
from datasets import load_metric
import argparse
import torch
import random



def preprocess_training_examples(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
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
        examples["context"],
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
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    metric = load_metric("squad")
    n_best = 20
    max_answer_length = 30
    example_to_features = collections.defaultdict(list)

    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            # For each sub-token returned by the tokenizer, the offset mapping gives us a tuple indicating the
            # sub-token’s start position and end position relative to the original token it was split from
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    try:
                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

                    except:
                        continue

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument('--squad_perc', default=75.0, type=float, help='The percentage of dataset to consider')
    parser.add_argument('--freq_eval', default=500, type=int, help="frequency w.r.t. global_step to perform evaluation")
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--checkpoints_dir', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str)


    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

        
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    raw_datasets = load_dataset("squad")
    if args.squad_perc:
        raw_datasets = {'train': raw_datasets['train'].select(range(int(args.squad_perc / 100 * len(raw_datasets['train'])))),
                        'validation': raw_datasets['validation'].select(range(int(args.squad_perc / 100 * len(raw_datasets['validation']))))}

    max_length = 384
    stride = 128

    train_dataset = raw_datasets["train"].map(
        (lambda examples: preprocess_training_examples(examples, tokenizer)),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
    (lambda examples: preprocess_validation_examples(examples, tokenizer)),
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
    )


    training_args = TrainingArguments(
        args.checkpoints_dir,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=args.freq_eval,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_dir=args.log_dir,            # directory for storing logs*
        logging_steps=10
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    
    # If add_argument is True the trainer will catch the last check-point available from the output-dir 
    if args.model_dir:
        print('Resuming training from checkpoint...')
        trainer.train(resume_from_checkpoint=args.checkpoints_dir)
    else:
        print("Starting fine-tuning from scratch... \n")
        trainer.train()

    predictions = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions[0]

    results = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
    print(results)
