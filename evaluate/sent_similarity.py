from sentence_transformers import SentenceTransformer, util
import torch

class Sent_Similar:
    def __init__(self, model='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'):
        self.evaluater = SentenceTransformer(model)
        self.evaluater.to(torch.device('cuda'))

    def get_scores(self, query, docs):
        #Encode query and documents
        query_emb = self.evaluater.encode(query)
        doc_emb = self.evaluater.encode(docs)

        #Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

        #Combine docs & scores
        doc_score_pairs = list(zip(docs, scores))

        #Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#         #Output passages & scores
#         for doc, score in doc_score_pairs:
#             print(score, doc)

        return scores, doc_score_pairs

    
    