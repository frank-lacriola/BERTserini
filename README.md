# BERTserini
This project is part of the Deep Natural Language Processing exam at Politecnico of Turin (academic year 2021-2022). 

The focus of the project is a develop a two-stages pipeline for Question-Answering. We base our work on [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718) paper. We also adapted the framework to use alternative models as readers, in particular the lightweight DistillBERT and RoBERTa. Moreover, we performed a small data enrichment addition training BERT model first on TriviaQA and then on SQUAD. Finally, a simple multi-lingual model is used in the pipeline to translate non-English questions to send to the retriever. The answer is then translated into the original language of the question. 

The scripts in this repository allow to finetune the model chosen on SQuAD and TriviaQA; to evaluate the two-stages pipeline on a subset of examples from SQuAD validation set; to ask a single question to the pipeline.

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
!python pipeline.py --model=<name or path to the model to load> --question_text=<text of the question to ask to the model> --k=10 --weight=0.5
```

# How to fine-tune on SQuAD
```
!python finetuning.py --squad_perc=80.0 --train_batch_size=16 --model_name=<name or path to the model> --checkpoints_dir=<checkpoint directory to resume a previously started training> --log_dir=<path to log directory> --model_dir=<directory of a pre-trained model>
```

# How to fine-tune on TriviaQA
```
!python finetune_triviaqa.py --checkpoints_dir=<checkpoint directory> --log_dir=<log directory> 
```

# How to evaluate the framework
```
!python evaluation.py --model=<name or path to the model>  --is_distill_bert=<boolean variable to use of distilBERT>  --num_eval_example=<the num of example you want to evaluate on>  --seed=<for reproducibility> --k=10 --weight=0.5
```

# Datasets
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- TriviaQA https://nlp.cs.washington.edu/triviaqa/


