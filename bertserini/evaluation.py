import argparse
from datasets import load_metric, load_dataset
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

	bertserini = BERTserini(model_to_load)

	predicted_answers=[]
	ground_truth = []
	for i,question in enumerate(val_examples):
	    print(question['question'])
	    bertserini.retrieve(question)
	    risposta = bertserini.answer()
	    predicted_answers.append(risposta)
	    ground_truth.append({'answers': val_examples['answers'][i], 'id':val_examples['id'][i]})
	result = metric.compute(predictions=predicted_answers, references=ground_truth)
	print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model to load")
    parser.add_argument('--num_eval_example', type=int, help="Number of examples to evaluate from the SQuAD validation set")
	parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args.model, args.num_eval_example)
