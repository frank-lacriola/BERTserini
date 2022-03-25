from reader import BERT

class BERTserini:
    def __init__(self, question_string, model_to_load):
        self.question = Question(question_string)
        self.searcher = LuceneSearcher.from_prebuilt_index("enwiki-paragraphs")
        self.bert = BERT(model_to_load)

    def retrieve(self):
        self.contexts = retrieve(self.question, self.searcher)
    
    def get_context(self):
        return self.contexts

    def answer(self):
        possible_questions = self.bert.predict(self.question, self.contexts)
        answer = get_best_answer(possible_questions, 0.45)
        return answer.text
    
    def get_question(self):
        return self.question.text
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model to load", required=True)
    parser.add_argument('question_text', type=str, required=True, help="Question to ask the model")


    args = parser.parse_args()

    bertserini = BERTserini(args.question_text ,args.model)
    bertserini.retrieve()
    risposta = bertserini.answer()
    print(risposta)