import re
import itertools
from verl.utils.reward_score.livebench.util import last_boxed_only_string, remove_boxed

def web_of_lies_v3_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    score = 0
    parsed_answer = None

    # extract text from <solution></solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    
    if len(solution_matches) > 0:
        parsed_answer = solution_matches[-1]

    # if parsed_answer is None:
    #     bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)
    #     if bold_words:
    #         bold_words = [word.lower().strip().replace(',', '').replace('.', '')[0:max(len(word), 3)] for match in bold_words for word in match.split()]

    #         parsed_answer = []
    #         i = len(bold_words) - 1
    #         while i >= 0 and len(parsed_answer) < 3:
    #             if bold_words[i] in ["yes", "no", "unknown"]:
    #                 parsed_answer = [bold_words[i]] + parsed_answer
    #             i -= 1

    #         if len(parsed_answer) > 0:
    #             parsed_answer = ", ".join(parsed_answer)
    #         else:
    #             parsed_answer = None

    allow_latex = True
    if parsed_answer is None or parsed_answer.strip() == '' and allow_latex:
        llm_answer = llm_answer.replace("\\\\boxed{\\\\textbf{", "\\\\boxed{")
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        llm_answer = llm_answer.replace("\\textbf{", "\\boxed{")

        last_boxed = last_boxed_only_string(llm_answer)

        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)

    allow_plain = True
    if allow_plain and parsed_answer is None:
        combs = itertools.product(['yes', 'no', 'unknown'], repeat=3)
        
        final_comb = None
        final_comb_index = -1
        for comb in combs:
            index = llm_answer.lower().find(', '.join(comb))
            if index != -1 and index > final_comb_index:
                final_comb = comb
                final_comb_index = index

        if final_comb is not None:
            parsed_answer = ', '.join(final_comb)

    if parsed_answer is None:
        score = 0
    else:
        parsed_answer_list = parsed_answer.split(',')
        ground_truth_list = ground_truth.split(',')

        num_correct = 0
        total = len(ground_truth_list)
        for i in range(total):
            if i >= len(parsed_answer_list):
                break
            gt_word = ground_truth_list[i]
            llm_word = parsed_answer_list[i]
            if llm_word == gt_word:
                num_correct += 1

        score = ((num_correct == total) + num_correct / total) / 2

        # score = 1 if parsed_answer.strip().replace(' ', '') == ground_truth.strip().replace(' ', '') else 0

    if debug and score < 1:
        print("INCORRECT")
        print('GROUND TRUTH', ground_truth)
        if parsed_answer:
            print('PARSED ANSWER', parsed_answer)
        else:
            print('PARSED ANSWER', parsed_answer)
        print('END OF OUTPUT', llm_answer[-50:])

    return score
