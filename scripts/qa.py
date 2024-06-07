from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

class QAModel:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        self.pipeline = None

    def set_pipeline(self, retriever):
        self.pipeline = ExtractiveQAPipeline(self.reader, retriever)

    def get_answer(self, question, context, top_k=1):
        answers = self.pipeline.run(query=question, documents=[{"text": context}], top_k_retriever=top_k, top_k_reader=top_k)
        if answers['answers']:
            return answers['answers'][0]['answer'], answers['answers'][0]['score']
        return "none", 0.0