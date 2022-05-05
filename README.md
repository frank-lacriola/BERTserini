# BERTserini

ABSTRACT

# Dependencies
- **Python 3** Test were made on Python 3.7.13
- [PyTorch](https://github.com/pytorch/pytorch) 1.11.0+cu113
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets) 
- [pyserini](https://github.com/castorini/pyserini) For the retriever
- [faiss](https://github.com/facebookresearch/faiss)
- [easyNMT](https://github.com/UKPLab/EasyNMT) For the translation of questions/answers


# How to run the QA Pipeline
Run this script to ask something to BERTserini and receive an answer:
```
!python pipeline.py --model=<name or path to the model to load> --question_text=<text of the question to ask to the model>
```

# How to fine-tune on SQuAD
```
!python finetuning.py 
```

# How to fine-tune on TriviaQA
```
!python finetune_triviaqa.py 
```

# How to evaluate the framework
```
!python evaluation.py --model=  --is_distill_bert=  --num_eval_example=  --seed=
```

# Datasets
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- TriviaQA https://nlp.cs.washington.edu/triviaqa/


