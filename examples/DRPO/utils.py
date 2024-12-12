import json
from search_algo import BeamSearchResult
from reasoners.algorithm import MCTSResult
from typing import Union

def parse_json_output(output: str) -> dict:
    # Strip leading/trailing whitespace and code block delimiters
    output = output.strip()
    output = output.strip('\n').strip('```').strip('JSON').strip('json')

    # Check if the output starts with a code block delimiter that includes a language specifier
    if output.startswith("```json"):
        # Remove the first line which contains ```json
        output = output.split('\n', 1)[1]
    
    # Strip the ending code block delimiter if present
    if output.endswith("```"):
        output = output.rsplit('```', 1)[0]

    # Further strip any leading/trailing whitespace that might be left
    output = output.strip()

    # Parse the json
    return json.loads(output, strict=False)

def parse_algo_output(algo_output: Union[BeamSearchResult, MCTSResult]) -> list[str]:
    trace = []
    if isinstance(algo_output, BeamSearchResult):
        for sub_result in algo_output.trace[-1][1]:
            trace.append(sub_result.system_prompt)
    
    elif isinstance(algo_output, MCTSResult):
        raise NotImplementedError("TODO")
    
    return trace