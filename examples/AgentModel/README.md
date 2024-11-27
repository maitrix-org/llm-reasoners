# Web Agent Experiment

## Setup

Please make sure that your environment has `opendevin` installed. If not, head to `web-agent-application` and run `pip install -e .`

## Commands

To run evaluation using one of our datasets, use the following command as an example:

```
python inference.py \
    [job_name] \
    --dataset [fanout, flight] \
    --start_idx 0 \
    --end_idx 20 \
    --output_dir browsing_data \
    [--use_world_model_planning]
```

One way to speed up the inference is to open several terminals and run inference on separate slices of the data.

Before that, you'll need to enter your default API key at `default_api_key.txt`


## Testing

To run on single examples for a quick test, you can use the following command: 

```
python main.py \
    [instruction] \
    [model] \
    [api_key]
```

## Log Visualizer

```
gradio log_visualizer/main.py
```

## Evaluation

```
python evaluation/fanout/run.py [job_name]
```