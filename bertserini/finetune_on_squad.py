import tensorflow_datasets as tfds
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import random
import argparse
import glob
import logging
import os
import random
import timeit
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


def save_ckpt(path, model, optimizer, scheduler, global_step, loss):
    """ save current model"""
    state = {
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "tr_loss": loss
    }
    torch.save(state, path)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, train_features, eval_features, eval_dataset, evaluation_examples, model, tokenizer):
    """ Train the model """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_sampler = RandomSampler(train_dataset)  # or DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.num_train_epochs  # giustificare

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        # not to do weight decay, VERIFICA
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    tr_loss, logging_loss = 0.0, 0.0

    # Train!
    global_step = 0
    steps_trained_in_current_epoch = 0

    # Loading states
    try:
        files = os.listdir(args.checkpoints_dir)
        if f'{args.checkpoints_filename}.pth' in files:
            checkpoint = torch.load(f'{args.checkpoints_dir}/{args.checkpoints_filename}.pth')
            print(f'Loading {args.checkpoints_filename}.pth from {args.checkpoints_dir}...')
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            steps_trained_in_current_epoch = checkpoint['global_step']
            check_loss = checkpoint['tr_loss']
            global_step = steps_trained_in_current_epoch
            tr_loss = check_loss
        else:
            print(f'No {args.checkpoints_filename}.pth in {args.checkpoints_dir}')
            print('Loading default model...')
    except:
        print('No chekpoint directory or file specified!')

    epochs_trained = 0
    model.zero_grad()
    # progress bar for epochs_trained
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    # Added here for reproductibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for _ in train_iterator:
        # progress bar for train_examplesw
        epoch_iterator = tqdm(train_dataloader)
        for step, batch in enumerate(epoch_iterator):

            # for amp optimization
            optimizer.zero_grad()

            # Skip past any already trained steps/batch if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_name == 'distilbert-base-uncased':
                del inputs['token_type_ids']

            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            scaler.scale(loss).backward()

            print("\n CURR LOSS: ", loss.item())
            tr_loss += loss.item()

            # EVALUATION:
            if args.do_eval and global_step != 0 and (global_step % 10 == 0 or global_step == (len(epoch_iterator) - 1)):
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                # Note that DistributedSampler samples randomly
                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

                # Eval!
                print("***** Running evaluation  *****")
                print("  Num examples = ", len(eval_dataset))
                print("  Batch size = ", args.eval_batch_size)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                all_results = []
                start_time = timeit.default_timer()

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    model.eval()
                    batch = tuple(t.to(device) for t in batch)

                    with torch.no_grad():

                        # During validation the labels are given in the examples and are predicted
                        inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }
                        feature_indices = batch[3]

                        if args.model_name == 'distilbert-base-uncased':
                            del inputs['token_type_ids']

                        # Casts operations to mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = model(**inputs)

                    for i, feature_index in enumerate(feature_indices):
                        eval_feature = eval_features[feature_index.item()]
                        unique_id = int(eval_feature.unique_id)
                        start_logits, end_logits = to_list(outputs.start_logits[i]), to_list(outputs.end_logits[i])

                        result = SquadResult(unique_id, start_logits, end_logits)

                        all_results.append(result)

                evalTime = timeit.default_timer() - start_time
                logger.info(f"  Evaluation done in total {evalTime} secs ({evalTime / len(eval_dataset)} sec per example")

                # Compute predictions and save output files
                output_dir = os.path.join(args.output_dir, f'{args.model_name}_{args.squad_perc}')
                if not os.path.exists(os.path.join(args.output_dir, f'{args.model_name}_{args.squad_perc}')):
                    os.makedirs(os.path.join(args.output_dir, f'{args.model_name}_{args.squad_perc}'))

                output_prediction_file = os.path.join(output_dir, "predictions.json")
                output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
                output_null_log_odds_file = None

                predictions = compute_predictions_logits(
                    evaluation_examples,
                    eval_features,
                    all_results,
                    args.n_best_size,
                    args.max_answer_length,
                    True,
                    output_prediction_file,
                    output_nbest_file,
                    output_null_log_odds_file,
                    args.verbose_logging,
                    False,
                    0,
                    tokenizer,
                )

                # Compute the F1 and exact scores.
                results = squad_evaluate(evaluation_examples, predictions)
                print("F1 score eval: ", results)
                # logging_loss = tr_loss

            # Unscales gradients and calls
            scaler.step(optimizer)
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            scaler.update()  # Updates the scale for next iteration
            global_step += 1

            print(tr_loss)
            try:
                if global_step != 0 and (global_step % 10 == 0 or global_step == (len(epoch_iterator) - 1)):
                    save_ckpt(f"{args.checkpoints_dir}/{args.checkpoints_filename}.pth", model, optimizer, scheduler,
                              global_step, tr_loss)
                    print(f' \n Checkpoint saved at {args.checkpoints_dir}/{args.checkpoints_filename}.pth')
            except:
                print('Checkpoint dir or file name wrong or not specified!')

            # To end epoch:
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # To end training:
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', default=0,
                        help='Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer')
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int, help="The maximum length of an answer that can be generated."
                        " This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action="store_true", help="If true, all of the warnings related to data processing will be printed.")
    parser.add_argument("--lang_id", default=0, type=int, help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)")
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--download_squad', action="store_true")
    parser.add_argument('--squad_perc', default=75.0, type=float, help='The percentage of dataset to consider')
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--checkpoints_filename', default=None, type=str)
    parser.add_argument('--checkpoints_dir', default=None, type=str)
    parser.add_argument('--squad_features_dir', default=None, type=str)


    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=True,
        use_fast=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name,
        config=config
    )

    model.to(device)

    squad = tfds.load('squad')

    if args.squad_perc:
        # Take only a limited num of examples for the training:
        print(f'You selected the {args.squad_perc}% of Squad dataset for fine-tuning \n')
        squad = {'train': squad['train'].take(int(args.squad_perc / 100 * len(squad['train']))),
                 'validation': squad['validation'].take(int(args.squad_perc / 100 * len(squad['validation'])))}

    processor = SquadV1Processor()
    training_examples = processor.get_examples_from_dataset(squad, evaluate=False)
    evaluation_examples = processor.get_examples_from_dataset(squad, evaluate=True)

    train_features = train_dataset = eval_features = eval_dataset = None
    if args.squad_features_dir:
        features_files = os.listdir(args.squad_features_dir)
        if (f'train_features_{args.squad_perc}.pt' in features_files
                and f'train_dataset_{args.squad_perc}.pt' in features_files):
            print(f'\n Loading squad train features and dataset from {args.squad_features_dir}...')
            train_features = torch.load(os.path.join(args.squad_features_dir, f'train_features_{args.squad_perc}.pt'))
            train_dataset = torch.load(os.path.join(args.squad_features_dir, f'train_dataset_{args.squad_perc}.pt'))

        if (f'eval_features_{args.squad_perc}.pt' in features_files
                and f'eval_dataset_{args.squad_perc}.pt' in features_files):
            eval_features = torch.load(os.path.join(args.squad_features_dir, f'eval_features_{args.squad_perc}.pt'))
            eval_dataset = torch.load(os.path.join(args.squad_features_dir, f'eval_dataset_{args.squad_perc}.pt'))
            print(f'\n Loading squad eval features and dataset from {args.squad_features_dir}...')

    if not (train_features and train_dataset and eval_features and eval_dataset):
        if not(train_features and train_dataset):
            train_features, train_dataset = squad_convert_examples_to_features(
                examples=training_examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                return_dataset="pt",
                # Either ‘pt’ or ‘tf’. if ‘pt’: returns a torch.data.TensorDataset, if ‘tf’: returns a tf.data.Dataset
                threads=1
            )
        if not(eval_features and eval_dataset):
            eval_features, eval_dataset = squad_convert_examples_to_features(
                examples=evaluation_examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                return_dataset="pt",
                # Either ‘pt’ or ‘tf’. if ‘pt’: returns a torch.data.TensorDataset, if ‘tf’: returns a tf.data.Dataset
                threads=1
            )
        if args.squad_features_dir:
            torch.save(train_features, os.path.join(args.squad_features_dir, f'train_features_{args.squad_perc}.pt'))
            torch.save(train_dataset, os.path.join(args.squad_features_dir, f'train_dataset_{args.squad_perc}.pt'))
            torch.save(eval_features, os.path.join(args.squad_features_dir, f'eval_features_{args.squad_perc}.pt'))
            torch.save(eval_dataset, os.path.join(args.squad_features_dir, f'eval_dataset_{args.squad_perc}.pt'))
            print(f'\n Squad features and datasets saved at {args.squad_features_dir}')

    print("Training/evaluation parameters %s", args)

    global_step, tr_loss = train(args, train_dataset, train_features, eval_features, eval_dataset,
                                 evaluation_examples, model, tokenizer)

    print(f" global_step = {global_step}, average loss = {tr_loss}")