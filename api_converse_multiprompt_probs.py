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
        Returns: full response, final answer, probability vector, format failure.
        Runs the prompt K times, computes empirical probabilities over d/r/u/l,
        and samples the final answer from a multinomial over those probabilities.
        format_failure is True only if ALL K responses failed to produce a valid answer.
        """
        K = self.config['num_samples']
        target_tokens = ['d', 'r', 'u', 'l']
        counts = np.zeros(4)  # d, r, u, l
        responses = []
        valid_count = 0

        print(colored(f"===== BEGIN K={K} BLOCK =====", 'light_grey'))
        for k in range(K):
            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
            )
            choice = response.choices[0]
            text = choice.message.content.strip()
            responses.append(text)

            # parse the final answer after the delimiter
            answer = None
            parts = text.split(delimiter)
            if len(parts) >= 2:
                answer = parts[-1].strip().lower()
                if len(answer) > 0:
                    answer = answer[0]
                if answer in target_tokens:
                    counts[target_tokens.index(answer)] += 1
                    valid_count += 1
                else:
                    answer = None

            print(colored(f"--- sample {k+1}/{K} ---", 'light_grey'))
            print(text)
            print(colored(f"    answer: {answer}", 'light_cyan'))

        # build concatenated full_response with markers
        full_response_parts = [f"===== BEGIN K={K} BLOCK ====="]
        for k, text in enumerate(responses):
            full_response_parts.append(f"--- sample {k+1}/{K} ---")
            full_response_parts.append(text)
            full_response_parts.append(f"--- end sample {k+1}/{K} ---")
        full_response_parts.append(f"===== END K={K} BLOCK =====")
        full_response = "\n".join(full_response_parts)
        print(colored(f"===== END K={K} BLOCK =====", 'light_grey'))

        # determine format failure, prob vector, and final answer
        if valid_count == 0:
            format_failure = True
            prob_vector = None
            final_answer = None
            print(colored(f"Warning: All {K} samples failed to produce a valid answer", 'light_red'))
        else:
            format_failure = False
            prob_vector = (counts / valid_count).tolist()
            # sample final answer from multinomial over empirical probabilities
            final_answer = np.random.choice(target_tokens, p=prob_vector)

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
        print(colored(f"Final answer: {final_answer}", 'light_green'))
        print(colored(f"Probabilities for [d,r,u,l]: {rounded_prob_vector}", 'light_magenta'))

        # Apply calibration if we have a calibrator and previous round probabilities
        calibrated_prob_vector = None
        if config.get('calibrator_models') is not None and prob_vector is not None and prev_prob_vector is not None:
            calibrated_prob_vector = calibrate_probabilities(
                config['calibrator_models'][r], prob_vector, prev_prob_vector
            )
            rounded_calibrated = [round(p, 4) for p in calibrated_prob_vector]
            print(colored(f"Calibrated probabilities for [d,r,u,l]: {rounded_calibrated}", 'light_cyan'))
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

        prompt = f"The other agent answered with probabilities for [d,r,u,l]: {rounded_probs_for_prompt}."

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
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="Number of conversation rounds between agents")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="Number of agents")
    parser.add_argument("--solo", type=str2bool, default=False,
                        help="Run solo full info baseline instead of collaborative game")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of times (K) to run each prompt for empirical probability estimation")
    parser.add_argument("--calibrator_path", type=str, default=None,
                        help="Path to directory containing trained calibrator models (e.g., calibrator_plots/)")
    parser.add_argument("--data_dir", type=str, default="out-api_exp1",
                        help="Directory containing input data")
    parser.add_argument("--out_dir", type=str, default="out-api_exp1/test",
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
        "num_samples": args.num_samples,
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
