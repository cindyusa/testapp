from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever

class Retriever:
    def __init__(self, host="localhost", port=9200, username="", password=""):
        self.document_store = ElasticsearchDocumentStore(host=host, port=port, username=username, password=password, index="document")
        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False
        )

    def write_documents(self, documents):
        self.document_store.write_documents(documents)

    def update_embeddings(self):
        self.document_store.update_embeddings(self.retriever)