from GPTScore.gpt3_score import gpt3score
from opt_score import directly_get_score

def evaluate_response(entailment_scorer, ctrleval_scorer, question, answer, knowledge):
    scores, _ = entailment_scorer.get_scores(question, [answer])
    entailment_score = scores[0]
    
    if knowledge:
        prefix = [knowledge]
        data = [knowledge+'\n'+answer]
        try:
            cons_result = ctrleval_scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1)
            cons_score = cons_result[0]
        except:
            cons_score = float('-inf')
    else:
        cons_score = float('-inf')
    return entailment_score, cons_score
        
    # print('cosistency', cons_result)
    
#     Pre, Recall, F1 = bert_score.score([response], [golden_response], lang="en", return_hash=False)
#     Pre = Pre.item()
#     Recall = Recall.item()
#     F1 = F1.item()
#     # print('bert_score Pre, Recall, F1', Pre, Recall, F1)

    

def evaluate_knowledge(gptscore_model, demo_num, question, knowledge, gptscore_tokenizer=None):
    PREFIX = {0: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.

Question: {question}
Knowledge: ''',
             1: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.

Question: What are the risk factors for heart disease?
Knowledge: Risk factors for heart disease can be categorized into modifiable and non-modifiable. Modifiable risk factors include high blood pressure, high cholesterol, smoking, unhealthy diet, physical inactivity, obesity, and excessive alcohol use. Non-modifiable risk factors include age, gender, family history, and race or ethnicity.
Question: {question}
Knowledge: ''',
             2: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.

Question: What are the risk factors for heart disease?
Knowledge: Risk factors for heart disease can be categorized into modifiable and non-modifiable. Modifiable risk factors include high blood pressure, high cholesterol, smoking, unhealthy diet, physical inactivity, obesity, and excessive alcohol use. Non-modifiable risk factors include age, gender, family history, and race or ethnicity.
Question: How does smoking affect lung health?
Knowledge: Smoking damages the airways and small air sacs in your lungs, which can lead to a variety of lung diseases including chronic bronchitis, emphysema, and lung cancer. It also decreases your lung capacity and makes it harder for your lungs to defend against infections and clear out mucus.
Question: {question}
Knowledge: ''',
             3: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.

Question: What are the risk factors for heart disease?
Knowledge: Risk factors for heart disease can be categorized into modifiable and non-modifiable. Modifiable risk factors include high blood pressure, high cholesterol, smoking, unhealthy diet, physical inactivity, obesity, and excessive alcohol use. Non-modifiable risk factors include age, gender, family history, and race or ethnicity.
Question: How does smoking affect lung health?
Knowledge: Smoking damages the airways and small air sacs in your lungs, which can lead to a variety of lung diseases including chronic bronchitis, emphysema, and lung cancer. It also decreases your lung capacity and makes it harder for your lungs to defend against infections and clear out mucus.
Question: Is it safe to take aspirin every day?
Knowledge: For some people, taking aspirin every day can help prevent heart attacks or strokes. However, daily aspirin isn't appropriate for everyone. It can cause side effects like gastrointestinal bleeding and isn’t recommended for people with certain health conditions or who take certain medications. Always consult with a healthcare professional before starting any new medication regimen.
Question: {question}
Knowledge: '''}
    prefix = PREFIX[demo_num]
    
    if gptscore_model == 'gpt3':
        gptscore = gpt3score(input=prefix, output=knowledge,
              gpt3model='davinci003',
              api_key="[YOUR API KEY]")
    else:
        srcs = [prefix]
        tgts = [knowledge]
        score_list = directly_get_score(gptscore_model, gptscore_tokenizer, srcs, tgts, prompt_text="")
        gptscore = score_list[0]
    return gptscore
    
