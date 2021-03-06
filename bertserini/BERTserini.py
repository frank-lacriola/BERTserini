from reader import BERT
import argparse
import numpy as np
from utils.structures import Question, Answer
from utils.utils_squad import get_best_answer
from pyserini.search.lucene import  LuceneSearcher
from easynmt import EasyNMT
from retriever import retrieve

class BERTserini:
    def __init__(self, model_to_load, is_distill_bert=False):
        self.searcher = LuceneSearcher.from_prebuilt_index("enwiki-paragraphs")
        self.bert = BERT(model_to_load, is_distilbert=is_distill_bert)
        self.lang_model = EasyNMT('opus-mt')


    def retrieve(self, example, k):
        self.original_question = example['question']
        self.lang = self.lang_model.language_detection(example['question'])
        print(f"Language detencted is: {self.lang} ")
        if self.lang != "en":
            tran = self.lang_model.translate(self.original_question, source_lang=self.lang, target_lang='en')
            print(f"Translated: {tran}")
            question = tran
        else:
            question = self.original_question
        self.question = Question(question)
        self.question.id = example['id']
        self.contexts = retrieve(self.question, self.searcher, k)
    
    def get_context(self):
        return self.contexts

    def answer(self, weight):
        possible_questions = self.bert.predict(self.question, self.contexts)
        answer = get_best_answer(possible_questions, weight)

        if self.lang !='en':
            print(f"Original answer: {answer.text}")
            tran = self.lang_model.translate(answer.text, source_lang='en', target_lang=self.lang)
            print("translated answer:", tran)
        pred_answer = {'prediction_text': answer.text, 'id': self.question.id}
        return pred_answer
    
    def get_question(self):
        return self.original_question
