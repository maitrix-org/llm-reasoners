from reasoners import Reasoner, GenerateOutput
from world_model import OrderQAWorldModel
from search_config import OrderQAConfig
from reasoners.lm import OpenAIAPIModel
from reasoners.algorithm import MCTS
from datetime import datetime
import argparse
import shutil
import csv
import os

class RunInstance:
    def __init__(self,
                 model_name="Meta-Llama-3.1-70B-Instruct",
                 questions_path="example_questions.csv", 
                 shots_path="zero_shot.txt",
                 check_answer_prompt_path="isAnswerCorrect.txt",
                 check_endstate_prompt_path="isEnd.txt",
                 fast_reward_prompt_path="fast_reward_prompt.txt",
                 reward_prompt_path="reward_prompt.txt",
                 check_answer_available_prompt_path="check_answer_available_prompt.txt",
                 extract_answer_prompt_path="extract_answer_prompt.txt",
                 output_dir="MCTS_output",
                 trace_number=0,
                 start_index=0,
                 end_index=-1):
        with open(f"questions/{questions_path}") as f:
            reader = csv.DictReader(f)
            self.questions = [{'question': row['Question'], 'answer': row['Answer']} for row in reader]
        
        with open(f"helper_prompts/{shots_path}") as f:
            self.shots = f.read().strip()
        
        with open(f"helper_prompts/{check_answer_prompt_path}") as f:
            self.check_answer_prompt = f.read().strip()
        
        with open(f"helper_prompts/{check_endstate_prompt_path}") as f:
            self.check_endstate_prompt = f.read().strip()
        
        with open(f"helper_prompts/{fast_reward_prompt_path}") as f:
            self.fast_reward_prompt = f.read().strip()
        
        with open(f"helper_prompts/{reward_prompt_path}") as f:
            self.reward_prompt = f.read().strip()
        
        with open(f"helper_prompts/{check_answer_available_prompt_path}") as f:
            self.check_answer_available_prompt = f.read().strip()
        
        with open(f"helper_prompts/{extract_answer_prompt_path}") as f:
            self.extract_answer_prompt = f.read().strip()
        
        self.trace_number=trace_number
        self.start_index=start_index
        self.end_index=end_index
        self.questions_name=questions_path.split('.')[0]
        
        self.llm_model = OpenAIAPIModel(model_name=model_name)
        self.world_model = OrderQAWorldModel(base_model=self.llm_model, 
                                             prompt=self.questions, 
                                             check_endstate_prompt=self.check_endstate_prompt)
        self.search_config = OrderQAConfig(base_model=self.llm_model,
                                           prompt=self.questions,
                                           check_endstate_prompt=self.check_endstate_prompt,
                                           check_answer_prompt=self.check_answer_prompt,
                                           fast_reward_prompt=self.fast_reward_prompt,
                                           reward_prompt=self.reward_prompt,
                                           check_answer_available_prompt=self.check_answer_available_prompt,
                                           extract_answer_prompt=self.extract_answer_prompt,
                                           trace_number=self.trace_number,
                                           QApairs=self.questions)
        self.search_algo = MCTS(output_trace_in_each_iter=True, depth_limit=1000)
        self.reasoner = Reasoner(world_model=self.world_model, search_config=self.search_config, search_algo=self.search_algo)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def isAnswerCorrect(self, answer, correct_answer, question):
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + question
        prompt_tail = self.check_answer_prompt  + '\n\n' + prompt_question
        prompt_answer = 'The student\'s answer is:\n' + answer
        prompt_correct_answer = 'The official correct answer is:\n' + correct_answer
        prompt_final = prompt_tail + '\n\n' + prompt_answer + '\n\n' + prompt_correct_answer
        # print(prompt_final)
        output = ""
        cnt = 0
        while output not in ["0", "1"]:
            model_output : GenerateOutput = self.llm_model.generate(temperature=0, prompts=[prompt_final])
            output = model_output.text[0]
            # print(f"isAnswerCorrectTrial: {output}")
            cnt+=1
            if cnt > 5:
                return -1
        # print(f"isAnswerCorrect: {int(output)}")
        return int(output)
        
    def run_single_question(self, question_number=0):
        shutil.rmtree("prompt_trace/", ignore_errors=True)
        
        algo_output = self.reasoner(example=self.shots, prompt=self.questions[question_number]['question'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/single_output_{timestamp}.txt"
        with open(filename, "w") as file:
            file.write(self.questions[question_number]['question']+'\n')
            for step in algo_output.terminal_state:
                file.write(f"{step}\n")

    def run_single_question_massive(self):
        run_count = 0
        output_data = []
        filename = f"{self.output_dir}/many_output.txt"
        start_time = datetime.now()
        while True:
            algo_output = self.reasoner(example=self.shots, prompt=self.questions[0]['question'])
            output_data.append(algo_output.terminal_state[-2])
            run_count += 1
            with open(filename, "a") as file:
                file.write(algo_output.terminal_state[-2] + "\n\n")
            print(f"Total Attempts: {run_count} Total Time: {datetime.now() - start_time}")
    
    def run_all_questions(self,
                          answer_per_question=1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.questions_name}/output_{timestamp}.csv"
        os.makedirs(f"{self.output_dir}/{self.questions_name}", exist_ok=True)
        
        # Open the CSV file in write mode initially to add the headers
        with open(filename, "w", newline='', encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["Index", "Question", "Reasoning", "Answer", "Correct answer", "isAnswerCorrect?"])
            writer.writeheader()
        
        # Loop Starts Here
        total_attempts = 0
        output_data = []
        start_time = datetime.now()
        prev_time = start_time
        
        # Determine the slice of questions
        if self.end_index == -1:
            question_slice = self.questions[self.start_index:]
        else:
            question_slice = self.questions[self.start_index:self.end_index + 1]
            
        for index, question_data in enumerate(question_slice, start=self.start_index):
            print(f"Question Index: {index}")
            for i in range(0, answer_per_question):
                # shutil.rmtree(f"step_traces{self.trace_number}/", ignore_errors=True)
                algo_output = self.reasoner(example=self.shots, prompt=question_data['question'])
                
                # from reasoners.visualization import visualize
                # from reasoners.visualization.tree_snapshot import NodeData, EdgeData
                # from reasoners.algorithm.mcts import MCTSNode

                # (Optional) You can write node_data_factory and edge_data_factory to show customized information.
                # def frq_node_data_factory(n: MCTSNode) -> NodeData:
                #     return NodeData({"state": n.state.blocks_state if n.state else "Not expanded",
                #                     "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
                #                     "# visited": len(n.cum_rewards)})

                # def frq_edge_data_factory(n: MCTSNode) -> EdgeData:
                #     return EdgeData({"Q": n.Q,
                #                     "intuition": n.fast_reward_details["intuition"],
                #                     "self_eval": n.fast_reward_details["self_eval"],
                #                     "action": n.action})

                # visualize(algo_output,
                #         node_data_factory=frq_node_data_factory,
                #         edge_data_factory=frq_edge_data_factory)
                
                reasoning = '\n'.join(algo_output.terminal_state)
                answer = algo_output.terminal_state[-1]
                correct_answer = question_data['answer']
                isAnswerCorrect = self.isAnswerCorrect(answer=answer, correct_answer=correct_answer, question=question_data['question'])
                question = question_data['question']
                attempt_data = {
                    "index": index,
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "correct answer": correct_answer,
                    "isAnswerCorrect": isAnswerCorrect
                }

                # Append attempt data to output_data
                output_data.append(attempt_data)
                total_attempts += 1

                # Append data to the file
                with open(filename, "a", newline='', encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=["index", "question", "reasoning", "answer", "correct answer", "isAnswerCorrect"])
                    writer.writerow(attempt_data)

                current_time = datetime.now()
                time_cost = round((current_time - prev_time).total_seconds(), 2)
                total_time = current_time - start_time
                formatted_total_time = f"{total_time.seconds // 3600}:{(total_time.seconds // 60) % 60:02}:{total_time.seconds % 60:02}.{round(total_time.microseconds / 10000):02}"
                prev_time = current_time

                print(f"# Attempt Of The Question: {i + 1} | Time Cost: {time_cost}s | Total Time: {formatted_total_time} | Total Attempts: {total_attempts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a instance with a specified question set.")
    parser.add_argument('-q', type=str, required=True, help='Question set name')
    parser.add_argument('-t', type=int, default=0, required=False, help='Trace file number')
    parser.add_argument('--start', type=int, default=0, required=False, help='Start question index')
    parser.add_argument('--end', type=int, default=-1, required=False, help='End question index (inclusive) -1 if you want to run til the file end')
    args = parser.parse_args()

    Instance = RunInstance(questions_path=f"{args.q}.csv",
                           trace_number=args.t,
                           start_index=args.start,
                           end_index=args.end,
                           shots_path="zero_shot_FRQ.txt",
                           check_endstate_prompt_path="isEnd_frq.txt")
    Instance.run_all_questions(answer_per_question=2)