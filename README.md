# BERTserini

ABSTRACT

# Dependencies
- **Python 3** Test were made on Python 3.7.13
- [PyTorch](https://github.com/pytorch/pytorch) 1.11.0+cu113
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets) 
- [pyserini](https://github.com/castorini/pyserini) For the retriever
- [easyNMT](https://github.com/UKPLab/EasyNMT) For the translation of questions/answers

# Datasets
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- TriviaQA https://nlp.cs.washington.edu/triviaqa/

# General QA Pipeline
Run this script to ask something to BERTserini and receive an answer:
```
!python pipeline.py --model=<name or path to the model to load> --question_text=<text of the question to ask to the model>
```

# Fine-tuning on SQuAD
```
!python finetuning.py 
```

# Fine-tuning on TriviaQA
```
!python finetune_triviaqa.py 
```

# Evaluation
```
!python evaluation.py --model=  --is_distill_bert=  --num_eval_example=  --seed=
```

