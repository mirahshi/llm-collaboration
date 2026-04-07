#!/usr/bin/env python3
"""
Collaborative maze conversation script using the ChatGPT API.
Reads the API key from gpt_api_key.txt, then runs multi-agent rounds.

    python api_converse.py
"""

from __future__ import annotations
from termcolor import colored
import os
import sys
from tqdm import tqdm

import re
import numpy as np
from pathlib import Path
from openai import OpenAI


class _Tee:
    """Write to a log file, and optionally also to the original stdout."""

    _ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, log_path: str, verbose: bool = True):
        self._stdout = sys.stdout
        self._verbose = verbose
        self._log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, data: str):
        if self._verbose:
            self._stdout.write(data)
        self._log.write(self._ansi_escape.sub("", data))

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def force_print(self, *args, **kwargs):
        """Always print to terminal and log, regardless of verbose setting."""
        import io
        buf = io.StringIO()
        print(*args, file=buf, **kwargs)
        text = buf.getvalue()
        self._stdout.write(text)
        self._log.write(self._ansi_escape.sub("", text))

    def close(self):
        sys.stdout = self._stdout
        self._log.close()

    # Delegate attribute lookups (e.g. `isatty`) to the real stdout.
    def __getattr__(self, name):
        return getattr(self._stdout, name)


def format_maze(maze_str):
    maze_size = int(len(maze_str) ** 0.5)
    formatted_maze = "\n".join(maze_str[i:i+maze_size] for i in range(0, len(maze_str), maze_size))
    return formatted_maze


class Agent():
    def __init__(self, config, round, id, client):
        self.config = config
        self.round = round
        self.id = id
        self.client = client

    def generate_response(self, prompt, delimiter='~'):
        import math
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.config['max_new_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            logprobs=True,
            top_logprobs=5,
        )
        choice = response.choices[0]
        full_response = choice.message.content.strip()

        prob_vector = None
        final_answer = None
        format_failure = False

        if choice.logprobs and choice.logprobs.content:
            token_contents = choice.logprobs.content
            delimiter_idx = None
            for i, token_info in enumerate(token_contents):
                if delimiter in token_info.token:
                    delimiter_idx = i
                    break
            
            if delimiter_idx is not None and delimiter_idx + 1 < len(token_contents):
                final_answer_token_info = token_contents[delimiter_idx + 1]
                final_answer = final_answer_token_info.token.strip()
                
                if config['append_probabilities']:
                    if not config['verbalize_probabilities']:
                        prob_vector = [0.0, 0.0, 0.0, 0.0]  # d, r, u, l
                        target_tokens = ['d', 'r', 'u', 'l']
                        for entry in final_answer_token_info.top_logprobs:
                            token = entry.token.strip().lower()
                            if token in target_tokens:
                                idx = target_tokens.index(token)
                                prob_vector[idx] += math.exp(entry.logprob)
                    else:
                        prob_match = re.search(r'%\s*\[([^\]]+)\]', full_response)
                        if prob_match:
                            prob_str = prob_match.group(1)
                            prob_vector = [float(p.strip()) for p in prob_str.split(',')]
                        if prob_vector is None or len(prob_vector) != 4:
                            print(colored(f"Warning: Probabilities for [d,r,u,l] should be 4 numbers, got {prob_vector}", 'light_red'))
                            prob_vector = None
                            format_failure = True
            else:
                print(colored(f"Warning: Delimiter '{delimiter}' not found or no token after delimiter", 'light_red'))
                format_failure = True

        return full_response, final_answer, prob_vector, format_failure


def api_converse(config, client, starting_prompts):
    # initialize agents
    agents = {
        agent_id: Agent(config, 0, agent_id, client)
        for agent_id in range(config['num_agents'])
    }
    # conversation loop
    prompt = ""
    for r in range(config['num_rounds']):
        print(colored(f"ROUND {r}: =================================================", 'light_yellow'))
        agent_id = r % config['num_agents']  # determine which agent acts this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(colored(f"PROMPT: {prompt}", 'light_blue'))
        full_response, final_answer, prob_vector, format_failure = agent.generate_response(prompt)
        
        if prob_vector is not None:
            rounded_prob_vector = [round(p, 4) for p in prob_vector]
        else:
            rounded_prob_vector = None
        print(f"Agent {agent_id} generated response: {full_response}")
        print(f"Final answer: {final_answer}")
        if config['append_probabilities']:
            print(f"Probabilities for [d,r,u,l]: {rounded_prob_vector}")
        
        if format_failure:
            return full_response, final_answer, True

        # update prompt for next round
        if config['append_probabilities']:
            prompt = f"The other agent said: {full_response} with probabilities for [d,r,u,l]: {rounded_prob_vector}."
        else:
            prompt = f"The other agent said: {full_response}."

    return full_response, final_answer, False

def generate_starting_prompt(config, maze_str, prefix, solo):
    formatted_maze = format_maze(maze_str)
    if not solo: # generate collaborative starting prompt
        return_string = f"""Task: You are going to play the collaborative maze game together with another agent. The maze consists of a grid with walls and a goal. Your task is to jointly determine the next move on the path. You and the other agent will take that action together. You will each get your own map of the same maze, with some coordinates hidden. Because of the hidden coordinates, you will need to communicate with the other agent to share information about the maze and coordinate your next move. The two agents' maps have complementary information. Both agents together have enough information to solve the maze, so you do not need to explore.
Rules:
- You can only move to adjacent cells [down(d), right(r), up(u), left(l)].
- You can not move diagonally or through walls.
- Once you make an incorrect move, you lose. 
Here is **your** map of the maze with a legend of the symbols: 
{formatted_maze} 
Legend:
@ - Starting Position at (0, 0) (before we have moved)
* - Goal Position
. - Path
# - Wall
? - Hidden Cell
"""
    else: # generate solo starting prompt
        return_string = f"""Task: You are going to play a maze game. The maze consists of a grid with walls and a goal. Your task is to navigate through the maze, avoiding walls, to reach the goal.
Rules:
- You can only move to adjacent cells [down(d), right(r), up(u), left(l)].
- You can not move diagonally or through walls.
- Once you make an incorrect move, you lose. 
Here is the map of the maze with a legend of the symbols: 
{formatted_maze} 
Legend:
@ - Starting Position at (0, 0) (before you have moved)
* - Goal Position
. - Path
# - Wall
"""

    if prefix != "": # add path prefix if it exists
        return_string += f"""
This is the path to the goal starting from @ that you have moved so far from which you can infer your current position: {prefix}
"""
    return_string += """
Explain your reasoning, then you must answer with one of d, r, u, l. Put your final answer after the delimiter ~. For example, if your final answer is d, your response should contain ~d. 
"""
    if config['append_probabilities'] and config['verbalize_probabilities']: # add instructions to verbalize probabilities
        return_string += """
Finally, give the probabilities you think each move d, r, u, l is correct after the delimiter %. For example, if you think the probability of d, r, u, l is 0.8, 0.1, 0.05, 0.05, your response should contain %[0.8, 0.1, 0.05, 0.05]. Make sure it is a valid probability vector, i.e. the sum of the probabilities is 1.
"""    
    return_string += """
Your moves are timed for speed so hurry up! 
"""
    return return_string



if __name__ == "__main__":
    # load API key
    api_key = Path("gpt_api_key.txt").read_text().strip()
    client = OpenAI(api_key=api_key)

    config = {
        "model_name": "gpt-4.1-mini",  # model to use for the conversation
        "max_new_tokens": 1024,  # allow for reasoning before final answer
        "temperature": 0.7,
        "top_p": 0.9,
        "num_rounds": 4,  # number of conversation rounds between agents
        "num_agents": 2,
        "solo": False, # if True, runs solo full info baseline; if False, runs collaborative game
        "append_probabilities": False, # if True, append probabilities to the output
        "verbalize_probabilities": True, # if False, probabilities taken from logprobs
        "data_dir": "out-api_exp1",
        "out_dir": "out-api_exp1/collab-no-probs",
        "verbose": False # print to terminal
    }

    # redirect all print output to out_dir/print_log.txt as well as stdout
    os.makedirs(config["out_dir"], exist_ok=True)
    _tee = _Tee(os.path.join(config["out_dir"], "print_log.txt"), verbose=config["verbose"])
    sys.stdout = _tee

    # run on dataset
    data_dir = config["data_dir"]
    success_count = 0
    format_failure_count = 0
    num_mazes = 50
    if config["solo"]:
        _tee.force_print(colored("Running solo full info baseline", 'light_yellow'))
        config["num_agents"] = 1
        config["num_rounds"] = 1
        input_file = os.path.join(data_dir, "input_full.txt")
        with open(input_file, "r") as f:
            input_lines = f.readlines()
        for i, input_line in tqdm(enumerate(input_lines[:num_mazes])):
            print(colored(f"Maze line {i}: ======================================================", 'light_magenta'))
            maze_str = input_line.split('=')[0].strip()
            label_sequence = input_line.split('=')[1].strip()

            prefix = ""
            # autoregessively generate the path
            for i in range(len(label_sequence)):
            # for i in range(3):
                print(colored(f"MOVE {i+1}: ======================================================", 'light_green'))
                starting_prompt = generate_starting_prompt(config, maze_str, prefix, solo=True)
                full_response, final_answer, format_failure = api_converse(config, client, [starting_prompt])
                if format_failure:
                    format_failure_count += 1
                    break
                if final_answer:
                    prefix += final_answer[0].lower()
                else:
                    prefix += full_response[-1]
                
                print(f"Updated prefix after move {i+1}: {prefix}")
            if format_failure:
                continue
            # Final evaluation of the generated path against the label sequence
            print(f"Final generated path: {prefix}")
            print(f"Label sequence: {label_sequence}")
            if prefix == label_sequence:
                success_count += 1
                print("Success! The generated path matches the label sequence.")
            else:
                print("The generated path does not match the label sequence.")
        _tee.force_print(f"Success rate: {success_count} / {num_mazes}")
        _tee.force_print(f"Format failure rate: {format_failure_count} / {num_mazes}")
    else:
        _tee.force_print(colored("Running collaborative conversation", 'light_yellow'))
        input_file0 = os.path.join(data_dir, "input0.txt")
        input_file1 = os.path.join(data_dir, "input1.txt")
        with open(input_file0, "r") as f0, open(input_file1, "r") as f1:
            input_lines0 = f0.readlines()
            input_lines1 = f1.readlines()
        for i, (input_line0, input_line1) in tqdm(enumerate(zip(input_lines0[:num_mazes], input_lines1[:num_mazes]))):
            print(colored(f"Maze line {i}: ======================================================", 'light_magenta'))
            maze_str0 = input_line0.split('=')[0].strip()
            maze_str1 = input_line1.split('=')[0].strip()
            label_sequence = input_line0.split('=')[1].strip()

            # maze_str0 = "@??....?#?#??.??..??.?.#.?.#..?.??#*"
            # maze_str1 = "@..????.?#?.#?#.??##?.???.????.?#.?*"
            # label_sequence = "rrrrrddlddrd"

            prefix = ""
            # autoregessively generate the path
            wrong_move = False
            for i in range(len(label_sequence)):
            # for i in range(3):
                print(colored(f"MOVE {i+1}: ======================================================", 'light_green'))
                starting_prompts = [generate_starting_prompt(config, maze_str0, prefix, solo=False), generate_starting_prompt(config, maze_str1, prefix, solo=False)]    
                full_response, final_answer, format_failure = api_converse(config, client, starting_prompts)
                prefix += final_answer[0].lower()
                if format_failure or final_answer is None:
                    print("Response failed")
                    format_failure_count += 1
                    break
                if final_answer[0].lower() != label_sequence[i]: # wrong move
                    print(f"Wrong move! The generated path {prefix} does not match the label sequence {label_sequence}.")
                    wrong_move = True
                    break
                print(f"Updated prefix after move {i+1}: {prefix}")
            
            if format_failure or wrong_move:
                continue
            print(f"Final generated path: {prefix}")
            print(f"Label sequence: {label_sequence}")
            success_count += 1
            print("Success! The generated path matches the label sequence.")
            # Final evaluation of the generated path against the label sequence
            # if prefix == label_sequence:
            #     success_count += 1
            #     print("Success! The generated path matches the label sequence.")
            # else:
            #     print("The generated path does not match the label sequence.")
        _tee.force_print(f"Success rate: {success_count} / {num_mazes}")
        _tee.force_print(f"Format failure rate: {format_failure_count} / {num_mazes}")

    _tee.close()

    # maze_str0 = "@#??#??#??.??.??.?.?.##?#..##?.#.??*"
    # # maze_str0 = "@##.#..##......#...#.##.#..###.#...*"
    # maze_str1 = "@?#.?..?#.?..?.#?.?#???.?????#???..*"

    # prefix = ""
    # label_sequence = "ddrrdddrrr"

    # # autoregessively generate the path
    # for i in range(len(label_sequence)):
    # # for i in range(3):
    #     print(colored(f"MOVE {i+1}: ======================================================", 'light_green'))
    #     starting_prompts = [generate_starting_prompt(config, maze_str0, prefix), generate_starting_prompt(config, maze_str1, prefix)]    
    #     full_response, final_answer = api_converse(config, client, starting_prompts)
    #     if final_answer:
    #         prefix += final_answer[0].lower()
    #     else:
    #         prefix += full_response[-1]
    #     print(f"Updated prefix after move {i+1}: {prefix}")

    # # Final evaluation of the generated path against the label sequence
    # print(f"Final generated path: {prefix}")
    # print(f"Label sequence: {label_sequence}")
    # if prefix == label_sequence:
    #     print("Success! The generated path matches the label sequence.")
    # else:
    #     print("The generated path does not match the label sequence.")
