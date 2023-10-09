from selfcheckgpt.modeling_selfcheck  import SelfCheckMQAG, SelfCheckBERTScore
import torch
import nlp

class SelfCheckGPT:
    def __init__(self, device='cuda', mqag=False, bertscore=False):
        if mqag:
            self.selfcheck_mqag = SelfCheckMQAG(device=device)
        if bertscore:
            self.selfcheck_bertscore = SelfCheckBERTScore(device=device)

    def get_sent_scores_mqag(self, passage, samples):
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sent_scores_mqag = self.selfcheck_mqag.predict(
            sentences = sentences,
            passage = passage,
            sampled_passages = samples,
            num_questions_per_sent = 5,
            scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
            beta1 = 0.8, beta2 = 0.8,)
        return sent_scores_mqag
    
    
    def get_sent_scores_bertscore(self, passage, samples):
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sent_scores_bertscore = selfcheck_bertscore.predict(
                sentences = sentences,
                sampled_passages = samples,)
        return sent_scores_bertscore