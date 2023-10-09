def main_loop(args, line, model, tokenizer, knowledge_loop, response_loop):
    all_history_knowledge, all_history_response = [], []
    
    THRESHOLD_ENTAIL = args.threshold_entailment
    MAX_LOOP = args.max_loop

    candidates = []
    main_loop_i = 0
    print(f"main_loop {main_loop_i}")
    question = line['question']

    if "generated_knowledge" in line.keys():
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question, [line['generated_knowledge']])
    else:
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question)
    all_history_knowledge += history_knowledge

    final_response, history_response, entailment_score_question = response_loop(args, model, tokenizer, question, final_knowledge)
    all_history_response += history_response
    candidates.append([entailment_score_question, final_knowledge, final_response])

    main_loop_i += 1
    while main_loop_i < MAX_LOOP and entailment_score_question < THRESHOLD_ENTAIL:
        print(f"main_loop {main_loop_i}")
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question)
        all_history_knowledge += history_knowledge

        final_response, history_response, entailment_score_question = response_loop(args, model, tokenizer, question, final_knowledge)
        all_history_response += history_response
        candidates.append([entailment_score_question, final_knowledge, final_response])
        main_loop_i += 1

    if (MAX_LOOP > 1) and entailment_score_question<THRESHOLD_ENTAIL:
        # still not satisified, highest_score
        candidates.sort()
        final_knowledge, final_response = candidates[-1][1:]
        
    return final_knowledge, final_response, all_history_knowledge, all_history_response
