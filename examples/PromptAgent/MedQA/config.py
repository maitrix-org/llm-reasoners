#config file
#hyperparameter
depth_limit=4
origin_prompt="Please use your domain knowledge in medical area to solve the questions."
num_batches=3
steps_per_gradient=1
batch_size=5
w_exp=2.5 
n_iters=12
# pre or pos prompt
prompt_position="pre"


from reasoners.lm import OpenAIModel
# model to answer questions
base_model = OpenAIModel(model="gpt-3.5-turbo", temperature=0)
# model to generate prompts and give feedback
optimize_model = OpenAIModel(model="gpt-4-turbo-preview", temperature=1)
