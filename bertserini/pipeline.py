from reader import BERT
import argparse
from utils.structures import Question, Answer
from pyserini.search.lucene import  LuceneSearcher
from retriever import retrieve

class BERTserini:
    def __init__(self, question_string, model_to_load):
        self.question = Question(question_string)
        self.searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")
        self.bert = BERT(model_to_load)

    def retrieve(self):
        self.contexts = retrieve(self.question, self.searcher)
    
    def get_context(self):
        return self.contexts

    def answer(self):
        possible_questions = self.bert.predict(self.question, self.contexts)
        for ans in possible_questions:
            ans.aggregate_score(0.80)
        best = sorted(possible_questions, key=lambda x: x.total_score, reverse=True)[0]
        return best.text
    
    def get_question(self):
        return self.question.text
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model to load")
    parser.add_argument('--question_text', type=str, help="Question to ask the model")


    args = parser.parse_args()

    bertserini = BERTserini(args.question_text ,args.model)
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
