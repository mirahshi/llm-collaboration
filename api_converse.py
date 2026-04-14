#!/usr/bin/env python3
"""
Collaborative maze conversation script using the ChatGPT API.
Reads the API key from gpt_api_key.txt, then runs multi-agent rounds.

    python api_converse.py
"""

from __future__ import annotations
from termcolor import colored
import argparse
import os
import sys
from tqdm import tqdm

import re
import numpy as np
from pathlib import Path
from openai import OpenAI
import math

from calibrator import load_calibrator, calibrate_probabilities


class _Tee:
    """Write to a log file, and optionally also to the original stdout."""

    _ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, log_path: str, verbose: bool = True):
        self._stdout = sys.stdout
        self._verbose = verbose
        self._log = open(log_path, "w", buffering=1, encoding="utf-8")

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
        """
        Returns: full response, final answer, probability vector, format failure
        format failure is True if the final answer is not in the target tokens or if the probabilities are not valid.
        """
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

        target_tokens = ['d', 'r', 'u', 'l']
        
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
                if final_answer not in target_tokens:
                    format_failure = True
                
                if self.config['verbalize_probabilities']: # get probabilities from response
                    prob_match = re.search(r'%\s*\[([^\]]+)\]', full_response)
                    if prob_match:
                        prob_str = prob_match.group(1)
                        prob_vector = [float(p.strip()) for p in prob_str.split(',')]
                    if prob_vector is None or len(prob_vector) != 4:
                        print(colored(f"Warning: Probabilities for [d,r,u,l] should be 4 numbers, got {prob_vector}", 'light_red'))
                        prob_vector = None
                        format_failure = True
                elif self.config['append_probabilities']: # get probabilities from logprobs
                    prob_vector = [0.0, 0.0, 0.0, 0.0]  # d, r, u, l
                    for entry in final_answer_token_info.top_logprobs:
                        token = entry.token.strip().lower()
                        if token in target_tokens:
                            idx = target_tokens.index(token)
                            prob_vector[idx] += math.exp(entry.logprob)
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

    # save per-round responses
    conversation_log = {
        'full_responses': [],
        'final_answers': [],
        'prob_vectors': [],
        'calibrated_prob_vectors': [],
        'format_failures': [],
    }

    # conversation loop
    prompt = ""
    prev_prob_vector = None
    for r in range(config['num_rounds']):
        print(colored(f"ROUND {r}: =================================================", 'light_yellow'))
        agent_id = r % config['num_agents']  # determine which agent acts this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(colored(f"PROMPT: {prompt}", 'light_blue'))
        full_response, final_answer, prob_vector, format_failure = agent.generate_response(prompt)

        # save responses for this round
        conversation_log['full_responses'].append(full_response)
        conversation_log['final_answers'].append(final_answer)
        conversation_log['prob_vectors'].append(prob_vector)
        conversation_log['format_failures'].append(format_failure)
        
        if prob_vector is not None:
            rounded_prob_vector = [round(p, 4) for p in prob_vector]
        else:
            rounded_prob_vector = None
        print(f"Agent {agent_id} generated response: {full_response}")
        print(f"Final answer: {final_answer}")
        if config['append_probabilities']:
            print(f"Probabilities for [d,r,u,l]: {rounded_prob_vector}")
        
        # Apply calibration if we have a calibrator and previous round probabilities
        calibrated_prob_vector = None
        if config.get('calibrator_models') is not None and prob_vector is not None and prev_prob_vector is not None:
            calibrated_prob_vector = calibrate_probabilities(
                config['calibrator_models'][r], prob_vector, prev_prob_vector
            )
            rounded_calibrated = [round(p, 4) for p in calibrated_prob_vector]
            print(f"Calibrated probabilities for [d,r,u,l]: {rounded_calibrated}")
        conversation_log['calibrated_prob_vectors'].append(calibrated_prob_vector)
        
        # stop the conversation if there is a format failure
        if format_failure:
            break

        # update prompt for next round
        # Use calibrated probabilities if available, otherwise use raw probabilities
        probs_for_prompt = calibrated_prob_vector if calibrated_prob_vector is not None else prob_vector
        if probs_for_prompt is not None:
            rounded_probs_for_prompt = [round(p, 4) for p in probs_for_prompt]
        else:
            rounded_probs_for_prompt = rounded_prob_vector

        if config['append_full_response']:
            prompt = f"The other agent said: {full_response}"
        else:
            if config['append_probabilities']: # append probabilities only
                prompt = f"The other agent answered with probabilities for [d,r,u,l]: {rounded_probs_for_prompt}."
            else: # append action only
                prompt = f"The other agent answered with: {final_answer}."

        # Track previous round's probabilities for calibration
        prev_prob_vector = prob_vector

    return conversation_log

def generate_starting_prompt(config, maze_str, prefix, solo):
    formatted_maze = format_maze(maze_str)
    if not solo: # generate collaborative starting prompt
        return_string = f"""Task: You are going to play the collaborative maze game together with another agent. The maze consists of a grid with walls and a goal. Your task is to jointly determine the next move on the path. You and the other agent will take that action together. You will each get your own map of the same maze, with some coordinates hidden. Because of the hidden coordinates, you will need to communicate with the other agent to share information about the maze and coordinate your next move. Both agents together have enough information to solve the maze, so you do not need to explore.
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
    if config['verbalize_probabilities']: # add instructions to verbalize probabilities
        return_string += """
Finally, give the probabilities you think each move d, r, u, l is correct after the delimiter %. For example, if you think the probability of d, r, u, l is 0.8, 0.1, 0.05, 0.05, your response should contain %[0.8, 0.1, 0.05, 0.05]. Make sure it is a valid probability vector, i.e. the sum of the probabilities is 1.
"""    
    return_string += """
Your moves are timed for speed so hurry up! 
"""
    return return_string



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false, yes/no, 1/0)')


def parse_args():
    parser = argparse.ArgumentParser(description="Collaborative maze conversation using ChatGPT API")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini",
                        help="Model to use for the conversation")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens for response (allow for reasoning before final answer)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_rounds", type=int, default=4,
                        help="Number of conversation rounds between agents")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="Number of agents")
    parser.add_argument("--solo", type=str2bool, default=False,
                        help="Run solo full info baseline instead of collaborative game")
    parser.add_argument("--append_probabilities", type=str2bool, default=False,
                        help="Communicate probabilities instead of actions")
    parser.add_argument("--verbalize_probabilities", type=str2bool, default=True,
                        help="Verbalize probabilities in prompt (if False, taken from logprobs)")
    parser.add_argument("--append_full_response", type=str2bool, default=False,
                        help="Communicate full response instead of just action/probabilities")
    parser.add_argument("--calibrator_path", type=str, default=None,
                        help="Path to trained calibrator models, e.g. out-api_exp2/calibrator")
    parser.add_argument("--data_dir", type=str, default="out-api_exp1",
                        help="Directory containing input data")
    parser.add_argument("--out_dir", type=str, default="out-api_exp1/test2",
                        help="Directory for output logs")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="Print to terminal")
    parser.add_argument("--start_maze", type=int, default=0,
                        help="Index of first maze (inclusive)")
    parser.add_argument("--end_maze", type=int, default=50,
                        help="Index of last maze (exclusive)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # load API key
    api_key = Path("gpt_api_key.txt").read_text().strip()
    client = OpenAI(api_key=api_key)

    # Load calibrators if specified
    calibrator_models = None
    if args.calibrator_path:
        calibrator_models = {}
        for r in range(1, args.num_rounds): # no calibrator for round 0
            calibrator_model, _, calibrator_round = load_calibrator(os.path.join(args.calibrator_path, f"calibrator_round{r}.pt"))
            print(f"Loaded calibrator from {os.path.join(args.calibrator_path, f'calibrator_round{r}.pt')} (trained on round {calibrator_round})")
            calibrator_models[r] = calibrator_model

    config = {
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_rounds": args.num_rounds,
        "num_agents": args.num_agents,
        "solo": args.solo,
        "append_probabilities": args.append_probabilities,
        "verbalize_probabilities": args.verbalize_probabilities,
        "append_full_response": args.append_full_response,
        "calibrator_models": calibrator_models,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "verbose": args.verbose,
    }
    start_maze = args.start_maze
    end_maze = args.end_maze
    num_mazes = end_maze - start_maze

    # redirect all print output to out_dir/print_log.txt as well as stdout
    os.makedirs(config["out_dir"], exist_ok=True)
    _tee = _Tee(os.path.join(config["out_dir"], "print_log.txt"), verbose=config["verbose"])
    sys.stdout = _tee

    # run on dataset
    data_dir = config["data_dir"]
    success_count = 0
    format_failure_count = 0

    conversations_dir = os.path.join(config["out_dir"], "conversations")
    os.makedirs(conversations_dir, exist_ok=True)

    maze_conversation_logs = {i: [] for i in range(start_maze, end_maze)} # save conversation logs for each maze
    if config["solo"]:
        _tee.force_print(colored("Running solo full info baseline", 'light_yellow'))
        config["num_agents"] = 1
        config["num_rounds"] = 1
        input_file = os.path.join(data_dir, "input_full.txt")
        with open(input_file, "r") as f:
            input_lines = f.readlines()
        for i, input_line in tqdm(enumerate(input_lines[start_maze:end_maze], start=start_maze)):
            print(colored(f"Maze line {i}: ======================================================", 'light_magenta'))
            maze_str = input_line.split('=')[0].strip()
            label_sequence = input_line.split('=')[1].strip()

            prefix = ""
            # autoregessively generate the path
            wrong_move = False
            for j in range(len(label_sequence)):
            # for i in range(3):
                print(colored(f"MOVE {j+1}: ======================================================", 'light_green'))
                starting_prompt = generate_starting_prompt(config, maze_str, prefix, solo=True)
                conversation_log = api_converse(config, client, [starting_prompt])

                # save label to conversation log
                conversation_log['label'] = label_sequence[j]
                
                # save conversation log for this maze
                maze_conversation_logs[i].append(conversation_log)
                
                # move onto next maze if there is a format failure
                format_failure = conversation_log['format_failures'][-1]
                if format_failure:
                    print("Response failed")
                    format_failure_count += 1
                    break
                
                final_answer = conversation_log['final_answers'][-1]
                prefix += final_answer[0].lower()

                # move onto next maze if the final answer is wrong
                if final_answer[0].lower() != label_sequence[j]:
                    print(f"Wrong move! The generated path {prefix} does not match the label sequence {label_sequence}.")
                    wrong_move = True
                    break
                print(f"Updated prefix after move {j+1}: {prefix}")
            
            # save conversation log for this maze
            np.save(os.path.join(conversations_dir, f"maze_{i}.npy"), {i: maze_conversation_logs[i]})

            if conversation_log['format_failures'][-1] or wrong_move:
                continue

            print(f"Final generated path: {prefix}")
            print(f"Label sequence: {label_sequence}")
            success_count += 1
            print("Success! The generated path matches the label sequence.")

        _tee.force_print(f"Success rate: {success_count} / {num_mazes}")
        _tee.force_print(f"Format failure rate: {format_failure_count} / {num_mazes}")
    else:
        _tee.force_print(colored("Running collaborative conversation", 'light_yellow'))
        input_file0 = os.path.join(data_dir, "input0.txt")
        input_file1 = os.path.join(data_dir, "input1.txt")
        with open(input_file0, "r") as f0, open(input_file1, "r") as f1:
            input_lines0 = f0.readlines()
            input_lines1 = f1.readlines()
        for i, (input_line0, input_line1) in tqdm(enumerate(zip(input_lines0[start_maze:end_maze], input_lines1[start_maze:end_maze]), start=start_maze)):
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
            for j in range(len(label_sequence)):
            # for i in range(3):
                print(colored(f"MOVE {j+1}: ======================================================", 'light_green'))
                starting_prompts = [generate_starting_prompt(config, maze_str0, prefix, solo=False), generate_starting_prompt(config, maze_str1, prefix, solo=False)]    
                conversation_log = api_converse(config, client, starting_prompts)

                # save label to conversation log
                conversation_log['label'] = label_sequence[j]
                
                # save conversation log for this maze
                maze_conversation_logs[i].append(conversation_log)
                
                # move onto next maze if there is a format failure
                format_failure = conversation_log['format_failures'][-1]
                if format_failure:
                    print("Response failed")
                    format_failure_count += 1
                    break
                
                final_answer = conversation_log['final_answers'][-1]
                prefix += final_answer[0].lower()

                # move onto next maze if the final answer is wrong
                if final_answer[0].lower() != label_sequence[j]:
                    print(f"Wrong move! The generated path {prefix} does not match the label sequence {label_sequence}.")
                    wrong_move = True
                    break
                print(f"Updated prefix after move {j+1}: {prefix}")
            
            # save conversation log for this maze
            np.save(os.path.join(conversations_dir, f"maze_{i}.npy"), {i: maze_conversation_logs[i]})

            if format_failure or wrong_move:
                continue

            print(f"Final generated path: {prefix}")
            print(f"Label sequence: {label_sequence}")
            success_count += 1
            print("Success! The generated path matches the label sequence.")

        _tee.force_print(f"Success rate: {success_count} / {num_mazes}")
        _tee.force_print(f"Format failure rate: {format_failure_count} / {num_mazes}")

    _tee.close()
