from typing import Optional, Mapping, Any


class Question:
    def __init__(self, text: str, id: Optional[str] = None, language: str = "en"):
        self.text = text
        self.id = id
        self.language = language


class Context:
    def __init__(self, id, score, text, language):
        self.id = id
        self.score = score
        self.text = text
        self.language = language


class Answer:
    def __init__(self,
                 text: str,
                 language: str = "en",
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 ctx_score: Optional[float] = 0,
                 total_score: Optional[float] = 0):
        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.ctx_score = ctx_score
        self.total_score = total_score
        
    def aggregate_score(self, weight):
        self.total_score = weight*self.score + (1-weight)*self.ctx_score