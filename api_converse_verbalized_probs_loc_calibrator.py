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
import re
import sys
from tqdm import tqdm

import numpy as np
from pathlib import Path
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types

from calibrator import load_calibrator, load_round_0_calibrator, calibrate_probabilities, calibrate_round_0_probabilities


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
        target_tokens = ['d', 'r', 'u', 'l']

        attempts_used = 0
        while attempts_used < 5:
            attempts_used += 1
            print(colored(f"Attempt {attempts_used} of 5", 'light_grey'))
            if self.config['api'] == 'claude':
                response = self.client.messages.create(
                    model=self.config['model_name'],
                    max_tokens=self.config['max_new_tokens'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config['temperature'],
                    thinking={"type": "adaptive"},
                    output_config={"effort": "low"},
                )
                full_response = next(
                    (b.text for b in response.content if b.type == "text"), ""
                ).strip()
            elif self.config['api'] == 'google':
                response = self.client.models.generate_content(
                    model=self.config['model_name'],
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self.config['max_new_tokens'],
                        temperature=self.config['temperature'],
                    ),
                )
                full_response = (response.text or "").strip()
            else:
                response = self.client.responses.create(
                    model=self.config['model_name'],
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature'],
                    reasoning={"effort": "low"},
                    # top_p=self.config['top_p'],
                    )
                full_response = response.output_text.strip()
            print(colored(f"Full response: {full_response}", 'light_grey'))

            prob_vector = None
            final_answer = None
            format_failure = False

            # Parse final answer from response string
            answer_match = re.search(re.escape(delimiter) + r'\s*([drul])', full_response, re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1).lower()
                if final_answer not in target_tokens:
                    format_failure = True
                    continue
            else:
                print(colored(f"Warning: Delimiter '{delimiter}' followed by a valid move not found in response", 'light_red'))
                format_failure = True
                continue
            
            # Parse probabilities from response string
            prob_match = re.search(r'%\s*\[([^\]]+)\]', full_response)
            if prob_match:
                prob_str = prob_match.group(1)
                try:
                    prob_vector = [float(p.strip()) for p in prob_str.split(',')]
                except Exception as e:
                    print(colored(f"Error parsing probabilities from string: {prob_str} ({e})", 'light_red'))
                    prob_vector = None
                    format_failure = True
                    continue
                
                if abs(sum(prob_vector) - 1.0) > 1e-6 or len(prob_vector) != len(target_tokens):
                    print(colored(f"Warning: Probabilities for [d,r,u,l] should be 4 numbers and sum to 1, got {prob_vector}", 'light_red'))
                    prob_vector = None
                    format_failure = True
                    continue
                break # valid response found
            else:
                print(colored(f"Warning: Probabilities for [d,r,u,l] should be 4 numbers, got {prob_vector}", 'light_red'))
                prob_vector = None
                format_failure = True
                continue

        print(colored(f"took {attempts_used} attempts" if final_answer else "failed to get a valid answer after 5 attempts", 'light_red'))

        return full_response, final_answer, prob_vector, format_failure


def api_converse(config, client, starting_prompts, prior_conversation_log=None):
    # initialize agents
    agents = {
        agent_id: Agent(config, 0, agent_id, client)
        for agent_id in range(config['num_agents'])
    }

    # save per-round responses
    if prior_conversation_log is None:
        conversation_log = {
            'full_responses': [],
            'final_answers': [],
            'prob_vectors': [],
            'calibrated_prob_vectors': [],
            'format_failures': [],
        }
    else: # use prior conversation log to continue the conversation
        conversation_log = prior_conversation_log

    if config['validation_mode']:
        end_round = config['end_round'] + 1 # at end_round, only save calibrated probs to conversation_log, do not generate responses
    else:
        end_round = config['end_round']
    for r in range(config['start_round'], end_round): 
        # conversation loop
        if r == 0:
            prompt = ""
            prev_prob_vector = None
        else:
            # Get the last calibrated probabilities from prior conversation log at round r if it exists
            # If not available, compute them using the calibrator for the previous round
            num_calibrated_rounds = len(conversation_log['calibrated_prob_vectors'])
            if num_calibrated_rounds >= r:
                prompt_probs = conversation_log['calibrated_prob_vectors'][r - 1]
            else:
                # Need to compute calibrated probs for the last round in prior log
                prev_round = r - 1
                if config.get('calibrator_models') is not None and prev_round in config['calibrator_models']:
                    # Get raw probs from prior log
                    current_raw = conversation_log['prob_vectors'][-1]  # p_{r-1} raw
                    print(colored(f"Base probs from round {prev_round}: {current_raw}", 'light_cyan'))
                    if prev_round == 0: # round 1
                        # apply calibrator for round 0
                        prompt_probs = calibrate_round_0_probabilities(
                            config['calibrator_models'][prev_round], current_raw
                        )
                    elif len(conversation_log['prob_vectors']) >= 2: # round 2 and beyond
                        if conversation_log['calibrated_prob_vectors'][-1] is not None:
                            prev_prev = conversation_log['calibrated_prob_vectors'][-1]
                            print(colored(f"Calibrated probs from round {prev_round - 1}: {prev_prev}", 'light_cyan'))
                        else:
                            prev_prev = conversation_log['prob_vectors'][-2]
                            print(colored(f"Base probs from round {prev_round - 1}: {prev_prev}", 'light_cyan'))
                        # if len(prior_conversation_log['calibrated_prob_vectors']) >= 2 and prior_conversation_log['calibrated_prob_vectors'][-1] is not None: 
                        #     prev_prev = prior_conversation_log['calibrated_prob_vectors'][-1]
                        #     print(colored(f"Calibrated probs from round {prev_round - 1}: {prev_prev}", 'light_cyan'))
                        # else:
                        #     prev_prev = prior_conversation_log['prob_vectors'][-2]
                        #     print(colored(f"Base probs from round {prev_round - 1}: {prev_prev}", 'light_cyan'))
                        # Apply calibrator for prev_round to get calibrated probs
                        prompt_probs = calibrate_probabilities(
                            config['calibrator_models'][prev_round], current_raw, prev_prev
                        )
                    # Round and renormalize
                    prompt_probs = [round(p, 2) for p in prompt_probs]
                    diff = round(1.00 - sum(prompt_probs), 2)
                    largest_idx = np.argmax(prompt_probs)
                    prompt_probs[largest_idx] = prompt_probs[largest_idx] + diff
                    prompt_probs = [round(p, 2) for p in prompt_probs]
                    print(colored(f"Applied calibrator_round{prev_round} to get calibrated probs: {prompt_probs}", 'light_cyan'))
                    # Save the computed calibrated probs back to conversation_log
                    conversation_log['calibrated_prob_vectors'].append(prompt_probs)
                else:
                    prompt_probs = conversation_log['prob_vectors'][-1]
                    print(colored(f"Base probs from round {prev_round}: {prompt_probs}", 'light_cyan'))
                    conversation_log['calibrated_prob_vectors'].append(None)
        
            prompt = f"The other agent answered with probabilities for [d,r,u,l]: {prompt_probs}."
            # Initialize prev_prob_vector from the (calibrated) probs so calibration is applied on first round
            prev_prob_vector = prompt_probs
        
        if r == config['end_round']:
            print(colored(f"Reached end_round {r}", 'light_yellow'))
            break
    
        print(colored(f"ROUND {r}: =================================================", 'light_yellow'))
        agent_id = r % config['num_agents']  # determine which agent acts this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(colored(f"PROMPT: {prompt}", 'light_blue'))
        full_response, final_answer, prob_vector, format_failure = agent.generate_response(prompt)

        
        
        if prob_vector is not None:
            rounded_prob_vector = [round(p, 2) for p in prob_vector]
            # renormalize to sum to 1
            diff = round(1.00 - sum(rounded_prob_vector), 2)
            largest_element_idx = np.argmax(rounded_prob_vector)
            rounded_prob_vector[largest_element_idx] = rounded_prob_vector[largest_element_idx] + diff
            renormalized_prob_vector = [round(p, 2) for p in rounded_prob_vector]
        else:
            renormalized_prob_vector = None
        print(colored(f"Final answer: {final_answer}", 'light_green'))
        print(colored(f"Probabilities for [d,r,u,l]: {renormalized_prob_vector}", 'light_magenta'))

        # # Apply calibration if we have a calibrator and previous round probabilities
        # calibrated_prob_vector = None
        # if config.get('calibrator_models') is not None and prob_vector is not None and prev_prob_vector is not None:
        #     calibrated_prob_vector = calibrate_probabilities(
        #         config['calibrator_models'][r], renormalized_prob_vector, prev_prob_vector
        #     )
        #     rounded_calibrated_prob_vector = [round(p, 2) for p in calibrated_prob_vector]
        #     # renormalize to sum to 1
        #     diff = round(1.00 - sum(rounded_calibrated_prob_vector), 2)
        #     largest_element_idx = np.argmax(rounded_calibrated_prob_vector)
        #     rounded_calibrated_prob_vector[largest_element_idx] = rounded_calibrated_prob_vector[largest_element_idx] + diff
        #     renormalized_calibrated_prob_vector = [round(p, 2) for p in rounded_calibrated_prob_vector]
        #     print(colored(f"Calibrated probabilities for [d,r,u,l]: {renormalized_calibrated_prob_vector}", 'light_cyan'))
        # else:
        #     renormalized_calibrated_prob_vector = None
        
        # save responses for this round
        conversation_log['full_responses'].append(full_response)
        conversation_log['final_answers'].append(final_answer)
        conversation_log['prob_vectors'].append(renormalized_prob_vector)
        # conversation_log['calibrated_prob_vectors'].append(renormalized_calibrated_prob_vector)
        conversation_log['format_failures'].append(format_failure)

        # stop the conversation if there is a format failure
        if format_failure:
            break

        # # update prompt for next round
        # # Use calibrated probabilities if available, otherwise use raw probabilities
        # probs_for_prompt = renormalized_calibrated_prob_vector if renormalized_calibrated_prob_vector is not None else renormalized_prob_vector

        # prompt = f"The other agent answered with probabilities for [d,r,u,l]: {probs_for_prompt}."

        # # Track previous round's probabilities for calibration
        # prev_prob_vector = probs_for_prompt

    return conversation_log

def generate_starting_prompt(config, maze_str, prefix="", solo=False):
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
@ - Current Position
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
@ - Current Position
* - Goal Position
. - Path
# - Wall
"""

    if prefix != "": # add path prefix if it exists
        return_string += f"""
This is the path to the goal that you have moved so far: {prefix}
"""
#     return_string += """
# Explain your reasoning, then you must answer with one of d, r, u, l. Put your final answer after the delimiter ~. For example, if your final answer is d, your response should contain ~d. 
# """
    return_string += """
You must answer with one of d, r, u, l. Put your final answer after the delimiter ~. For example, if your final answer is d, your response should contain ~d. 
"""
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
    parser = argparse.ArgumentParser(description="Collaborative maze conversation using OpenAI, Claude, or Google API")
    parser.add_argument("--api", type=str, default="openai", choices=["openai", "claude", "google"],
                        help="Backend API to use: 'openai' (default), 'claude', or 'google'")
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
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of times (K) to run each prompt for empirical probability estimation")
    parser.add_argument("--calibrator_path", type=str, default=None,
                        help="Path to directory containing trained calibrator models (e.g., calibrator_plots/)")
    parser.add_argument("--data_dir", type=str, default="out-api_exp1",
                        help="Directory containing input data")
    parser.add_argument("--out_dir", type=str, default="out-api_exp1/multiprompt_probs",
                        help="Directory for output logs")
    parser.add_argument("--start_maze", type=int, default=0,
                        help="Index of first maze (inclusive)")
    parser.add_argument("--end_maze", type=int, default=20,
                        help="Index of last maze (exclusive)")
    parser.add_argument("--start_round", type=int, default=0,
                        help="Index of first round (inclusive)")
    parser.add_argument("--end_round", type=int, default=4,
                        help="Index of last round (exclusive)")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="Print to terminal")
    parser.add_argument("--hard_input_data_dir", type=str, default=None,
                        help="If specified, use hard input data")     
    parser.add_argument("--validation_mode", type=str2bool, default=False,
                        help="Run in validation mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # build client based on selected backend
    if args.api == "claude":
        api_key = Path("anthropic_api_key.txt").read_text().strip()
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Loaded Claude client for model {args.model_name}")
    elif args.api == "google":
        api_key = Path("google_api_key.txt").read_text().strip()
        client = genai.Client(api_key=api_key)
        print(f"Loaded Google client for model {args.model_name}")
    else:
        api_key = Path("gpt_api_key.txt").read_text().strip()
        client = OpenAI(api_key=api_key)
        print(f"Loaded OpenAI client for model {args.model_name}")

    # Load calibrators if specified
    calibrator_models = None
    if args.calibrator_path:
        calibrator_models = {}
        if args.validation_mode:
            start = args.start_round
            end = args.end_round
        else:
            start = args.start_round-1
            end = args.start_round
        for r in range(start, end): 
            if r == 0:
                if os.path.exists(os.path.join(args.calibrator_path, f"calibrator_round{r}.pt")):
                    calibrator_model, _, calibrator_round = load_round_0_calibrator(os.path.join(args.calibrator_path, f"calibrator_round{r}.pt"))
                    calibrator_models[r] = calibrator_model
            else:
                calibrator_model, _, calibrator_round = load_calibrator(os.path.join(args.calibrator_path, f"calibrator_round{r}.pt"))
                print(f"Loaded calibrator from {os.path.join(args.calibrator_path, f'calibrator_round{r}.pt')} (trained on round {calibrator_round})")
                calibrator_models[r] = calibrator_model

    start_round = args.start_round
    end_round = args.end_round

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
        "start_round": start_round,
        "end_round": end_round,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "verbose": args.verbose,
        "api": args.api,
        "hard_input_data_dir": args.hard_input_data_dir,
        "validation_mode": args.validation_mode,
    }
    start_maze = args.start_maze
    end_maze = args.end_maze
    num_mazes = end_maze - start_maze

    os.makedirs(config["out_dir"], exist_ok=True)

    # run on dataset
    data_dir = config["data_dir"]
    hard_data_dir = config["hard_input_data_dir"]
    # get mazes from data_dir
    # mazes is a dictionary mapping maze index to maze strings
    input_mazes0 = {}
    input_mazes1 = {}
    with open(os.path.join(data_dir, "input0.txt"), "r") as f0, open(os.path.join(data_dir, "input1.txt"), "r") as f1:
        input_lines0 = f0.readlines()
        input_lines1 = f1.readlines()
    if hard_data_dir is not None:
        with open(os.path.join(hard_data_dir, "hard_input0.txt"), "r") as f0, open(os.path.join(hard_data_dir, "hard_input1.txt"), "r") as f1:
            hard_input_lines0 = f0.readlines()
            hard_input_lines1 = f1.readlines()
    maze_starting_indices = [i for i, example in enumerate(input_lines0) if example[0] == '@']
    maze_starting_indices = maze_starting_indices[start_maze:end_maze+1] 
    num_mazes_retrieved = len(maze_starting_indices) - 1 # -1 because we don't want to include the last maze starting index
    print(f"Number of mazes retrieved: {num_mazes_retrieved}")
    for i, maze_index in enumerate(range(start_maze, end_maze)):
        if hard_data_dir is not None: # only include input lines that are in the hard dataset
            input_mazes0[maze_index] = [input_lines0[j].strip() for j in range(maze_starting_indices[i], maze_starting_indices[i+1]) if input_lines0[j] in hard_input_lines0]
            input_mazes1[maze_index] = [input_lines1[j].strip() for j in range(maze_starting_indices[i], maze_starting_indices[i+1]) if input_lines1[j] in hard_input_lines1]
        else:
            input_mazes0[maze_index] = [input_lines0[j].strip() for j in range(maze_starting_indices[i], maze_starting_indices[i+1])]
            input_mazes1[maze_index] = [input_lines1[j].strip() for j in range(maze_starting_indices[i], maze_starting_indices[i+1])]
    print(f"Total number of input lines: {sum([len(input_mazes0[maze_idx]) for maze_idx in range(start_maze, end_maze)])}")
    
    success_count = 0
    format_failure_count = 0

    if config['validation_mode']:
        conversations_dir = os.path.join(config["out_dir"], f"conversations{config['start_round']}-{config['end_round']-1}")
    else:
        conversations_dir = os.path.join(config["out_dir"], f"conversations{config['start_round']}")
    os.makedirs(conversations_dir, exist_ok=True)

    if start_round > 0:
        conversations_dir_prior = os.path.join(config["out_dir"], f"conversations{config['start_round'] - 1}")
        print(f"Path to saved conversations from prior rounds: {conversations_dir_prior}")
    else:
        conversations_dir_prior = None

    maze_conversation_logs = {i: [] for i in range(start_maze, end_maze)} # save conversation logs for each maze
    if config["solo"]:
        print(colored("Running solo full info baseline", 'light_yellow'))
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

        print(f"Success rate: {success_count} / {num_mazes}")
        print(f"Format failure rate: {format_failure_count} / {num_mazes}")
    else:
        print(colored("Running collaborative conversation", 'light_yellow'))
        input_file0 = os.path.join(data_dir, "input0.txt")
        input_file1 = os.path.join(data_dir, "input1.txt")
        with open(input_file0, "r") as f0, open(input_file1, "r") as f1:
            input_lines0 = f0.readlines()
            input_lines1 = f1.readlines()
        
        mazes_with_format_failure = 0
        for i in range(start_maze, end_maze):
            print(colored(f"Maze {i}: ======================================================", 'light_magenta'))
            maze_lines0 = input_mazes0[i]
            maze_lines1 = input_mazes1[i]

            # Skip mazes with no hard moves (before attempting to load prior logs)
            if len(maze_lines0) == 0:
                print(colored(f"Skipping maze {i} - no hard moves", 'light_grey'))
                continue

            if start_round > 0: # get conversation logs for this maze with data from previous rounds
                prior_conversation_logs = np.load(os.path.join(conversations_dir_prior, f"maze_{i}.npy"), allow_pickle=True).item()

            # track success for this maze
            success = True
            maze_had_format_failure = False
            for j in range(len(maze_lines0)):
                print(colored(f"MOVE {j+1}: ======================================================", 'light_green'))
                maze_str0 = maze_lines0[j].split('=')[0].strip()
                maze_str1 = maze_lines1[j].split('=')[0].strip()
                label = maze_lines0[j].split('=')[1].strip()
                # correct_prefix = "".join([line.split('=')[1].strip() for line in maze_lines0[0:j]])
                starting_prompts = [generate_starting_prompt(config, maze_str0, solo=False), generate_starting_prompt(config, maze_str1, solo=False)]    

                if start_round == 0:
                    prior_conversation_log = None
                else:
                    prior_conversation_log = prior_conversation_logs[i][j] 
                conversation_log = api_converse(config, client, starting_prompts, prior_conversation_log)

                # save label to conversation log
                conversation_log['label'] = label
                
                # save conversation log for this maze
                maze_conversation_logs[i].append(conversation_log)
                
                # track last round format failure - continue to next move
                format_failure = conversation_log['format_failures'][-1]
                if format_failure:
                    print("Final response failed")
                    maze_had_format_failure = True
                    success = False
                    continue
                
                final_answer = conversation_log['final_answers'][-1]

                # track last round wrong move - continue to next move
                if final_answer[0].lower() != label:
                    print(f"Wrong move! The generated move {final_answer} does not match the label {label}.")
                    success = False
                    continue
            
            # save conversation log for this maze if there are any conversation logs
            if len(maze_conversation_logs[i]) > 0:
                np.save(os.path.join(conversations_dir, f"maze_{i}.npy"), {i: maze_conversation_logs[i]})

            if maze_had_format_failure:
                mazes_with_format_failure += 1

            if success:
                success_count += 1
                print("Success! The generated path matches the label sequence.")
            else:
                print("Failure! The generated path does not match the label sequence.")

        print(f"Success rate: {success_count} / {num_mazes}")
        print(f"Format failure rate: {mazes_with_format_failure} / {num_mazes}")
