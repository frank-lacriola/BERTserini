import argparse
from datasets import load_metric
from BERTserini import BERTserini
import numpy as np

def get_validation_examples(num_eval_example):
	squad = load_dataset("squad")

	indx = np.random.uniform(low=0,high=squad['validation'].num_rows,size=num_eval_example)
	val_examples = squad['validation'].select(indx)
	return val_examples


def main(model_to_load, num_val_examples):
	metric = load_metric("squad")
	
	val_examples = get_validation_examples(num_val_examples)

	predicted_answers=[]
	ground_truth = []
	for question in val_examples:
	    print(question)
	    bertserini = BERTserini(question, model_to_load)
	    bertserini.retrieve()
	    risposta = bertserini.answer()
	    predicted_answers.append(risposta)
	    ground_truth.append({'answers': val_examples['answers'][0], 'id':val_examples['id'][0]})
	result = metric.compute(predictions=predicted_answers, references=ground_truth)
	print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model to load")
    parser.add_argument('--num_eval_example', type=int, help="Number of examples to evaluate from the SQuAD validation set")

    args = parser.parse_args()

    main(args.model, args.num_eval_example)