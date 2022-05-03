from BERTserini import BERTserini
import argparse
import numpy as np

def main(args):
    example_question = {'question':args.question_text, "id":np.random.randint(0,10545714)}

    bertserini = BERTserini(example_question,args.model)
    print("*"*60)
    print("Retrieving contexts...")
    print("*"*60)
    bertserini.retrieve()
    print("*"*60)
    print("Finding answering...")
    print("*"*60)
    risposta = bertserini.answer()
    print("*"*60)
    print("Answer:")
    print(risposta)
    print("*"*60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model to load")
    parser.add_argument('--question_text', type=str, help="Question to ask the model")

    args = parser.parse_args()

    main(args)