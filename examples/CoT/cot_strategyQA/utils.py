def extract_final_answer(sample):
    ans = ''
    if "So the answer is" in sample:
        ## extract answer directly
        # ans_idx = sample.find("So the answer is")
        # ans = re.findall(r'\d+', sample[ans_idx:])
        ans = sample.split('So the answer is')
        if ans:
            ans = ans[-1].strip().split('\n')[0].replace('.', '')
        else:
            ans = ''
    else:
        ## negative word list
        if ' not ' in sample or ' no ' in sample or 'Not ' in sample or 'No ' in sample:
            ans = 'no'
            # print(f"find no: {ans}, {sample}")
        else:
            ans = ''
    return ans

        
def eval_output(answer, output):
    if output is None:
        return False
    
    # False vs no and True vs yes
    answer = "no" if not answer else "yes"
    
    return answer == output.strip().lower()