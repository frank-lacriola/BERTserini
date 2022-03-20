from utils.structures import Context


def retrieve(question, searcher):
    hits = searcher.search(question.text)
    contexts = []
    for i in range(0, len(hits)):
        id = hits[i].docid
        print(f"id: {id}")
        score = hits[i].score
        document = searcher.doc(hits[i].docid)
        #document = json.loads(document.raw()) # for indexes 'wikipedia-dpr'
        text = document.raw()               # for indexes 'enwiki-paragraphs'
        print(document.raw())
        #text = document['contents']
        language = 'en'
        #print(document['contents'])
        para = Context(id, score, text, language)
        contexts.append(para)
    return contexts