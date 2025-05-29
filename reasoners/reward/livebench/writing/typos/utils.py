import difflib
import re

def extract_answer(llm_answer):
    pattern = r'.* --- (.*?) --- .*'
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer

def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:

    parsed_answer = None

    # extract text from <solution></solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if len(solution_matches) > 0:
        match = solution_matches[-1]
        parsed_answer = match
    else:
        parsed_answer = llm_answer.replace('<solution>', '').replace('</solution>', '')
        parsed_answer = extract_answer(parsed_answer)

    parsed_answer = ' '.join(list(filter(None, parsed_answer.strip().split('\n'))))

    if int(ground_truth in parsed_answer):
        return 1

    score = 0

    if debug and score == 0:

        a = ground_truth
        b = parsed_answer
        m = difflib.SequenceMatcher(a=a, b=b)
        pad = 10

        for tag, i1, i2, j1, j2 in m.get_opcodes():
            length = min(len(parsed_answer), len(ground_truth))
            mi1, mi2, mj1, mj2 = max(0,i1-pad), min(length, i2+pad), max(0, j1-pad), min(length, j2+pad)

            mistake_length = 0
            if tag == 'replace':
                print("<changed>", a[i1:i2], b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
                mistake_length = i2 - i1
            if tag == 'delete':
                print("<deleted>", a[i1:i2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
                mistake_length = i2 - i1
            if tag == 'insert':
                print("<inserted>", b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
                mistake_length = j2 - j1
            #score = score - (mistake_length / len(ground_truth)) * 100


    if debug and score == 0:
        print('INCORRECT')
        print('GROUND TRUTH', ground_truth)
        print('SOLUTION', parsed_answer)
        print('END OF OUTPUT', llm_answer[-len(parsed_answer):])

    return score
