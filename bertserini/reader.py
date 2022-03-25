from transformers import AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features
from torch.utils.data import DataLoader, SequentialSampler
import torch
from transformers.data.processors.squad import SquadResult
from utils.structures import  Answer
from finetune_on_squad import to_list
from utils.utils_squad import SquadExample, compute_predictions_logits


class BERT:
    def init(self, model_name: str = 'bert-base-uncased', tokenizer_name: str = 'bert-base-uncased', checkpoints_dir):
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.args = {
            "max_seq_length": 384,
            "doc_stride": 128,
            "max_query_length": 64,
            "threads": 1,
            "tqdm_enabled": False,
            "n_best_size": 20,
            "max_answer_length": 30,
            "do_lower_case": True,
            "output_prediction_file": False,
            "output_nbest_file": None,
            "output_null_log_odds_file": None,
            "verbose_logging": False,
            "version_2_with_negative": True,
            "null_score_diff_threshold": 0,
        }
        self.checkpoints_dir = checkpoints_dir
        
    def predict(self, question, contexts):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
        state_dict = torch.load(self.checkpoints_dir)
        model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
        model.load_state_dict(state_dict['model_state'])
        model.to(device)

        # Let's create examples from contexts to feed the model
        examples = []
        for idx, ctx in enumerate(contexts):
            examples.append(
                SquadExample(
                    qas_id=idx,
                    question_text=question.text,
                    context_text=ctx.text,
                    answer_text=None,
                    start_position_character=None,
                    title="",
                    is_impossible=False,
                    answers=[],
                    language=ctx.language
                )
            )

        features, dataset = squad_convert_examples_to_features(
                  examples=examples,
                  tokenizer=tokenizer,
                  max_seq_length=384,
                  doc_stride=128,
                  max_query_length=64,
                  is_training=False,
                  return_dataset= "pt", # Either ‘pt’ or ‘tf’. if ‘pt’: returns a torch.data.TensorDataset, if ‘tf’: returns a tf.data.Dataset
                  threads= 1
                )

        eval_sampler = SequentialSampler(dataset)  # A Sampler that returns indices sequentially.
        eval_dataset = DataLoader(dataset, sampler=eval_sampler, batch_size=32)

        all_results = []
        for batch in eval_dataset:
            model.eval() # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    # The attention mask is an optional argument used when batching sequences together
                    # This argument indicates to the model which tokens should be attended to, and which should not.
                    "attention_mask": batch[1],
                    # Other models, such as BERT, also deploy token type IDs (also called segment IDs).
                    # They are represented as a binary mask identifying the two types of sequence in the model
                    "token_type_ids": batch[2],
                }
                feature_indices = batch[3]
                outputs = model(**inputs)
                print(outputs)
                #print(f"start: {outputs[0]}")
                #print(f"end: {outputs[1]}")
                start_logits, end_logits = outputs[0], outputs[1]
                for i, feature_index in enumerate(feature_indices):
                        eval_feature = features[feature_index.item()]
                        unique_id = int(eval_feature.unique_id)
                        start_logits, end_logits = to_list(outputs.start_logits[i]), to_list(outputs.end_logits[i])
                        #print(start_logits, end_logits)
                        result = SquadResult(unique_id, start_logits, end_logits)
                        all_results.append(result)

        answers, n_best = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=20,
            max_answer_length=64,
            do_lower_case=True,
            output_prediction_file=False,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=1,
            version_2_with_negative=False,
            null_score_diff_threshold=0,
            tokenizer=tokenizer
        )

        all_answers = []
        for idx, ans in enumerate(n_best):
            all_answers.append(Answer(
                text=answers[ans][0],
                score=answers[ans][1],
                ctx_score=contexts[idx].score,
                language='en'
            ))

        return all_answers