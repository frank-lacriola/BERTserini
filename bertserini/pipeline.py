from BERTserini import BERTserini
import argparse
import numpy as np

def main(args):
    example_question = {'question': args.question_text, "id": np.random.randint(0,10545714)}

    bertserini = BERTserini(args.model)
    print("*" * 60)
    print("Retrieving contexts...")
    print("*" * 60)
    bertserini.retrieve(example_question, k=args.k)
    print("*"*60)
    print("Finding answering...")
    print("*"*60)
    risposta = bertserini.answer(args.weight)
    print("*"*60)
    print("Answer:")
    print(risposta)
    print("*"*60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="name or path to model to load")
    parser.add_argument('--k', type=int, default=10, help="the number of context to retrieve")
    parser.add_argument('--weight', type=str, default=0.5, help="weight for the linear interpolation")
    parser.add_argument('--question_text', type=str, help="Question to ask the model")

    args = parser.parse_args()

    main(args)