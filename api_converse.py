#!/usr/bin/env python3
"""
Collaborative maze conversation script using the ChatGPT API.
Reads the API key from gpt_api_key.txt, then runs multi-agent rounds.

    python api_converse.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from openai import OpenAI


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

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            logprobs=True,
            top_logprobs=5,
        )
        print(len(response.choices))
        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            token_info = choice.logprobs.content[0]
            print(f"Token: '{token_info.token}'")
            for entry in token_info.top_logprobs:
                print(f"  '{entry.token}': {np.exp(entry.logprob):.4f}")
        
        choice = response.choices[0]
        argmax_token = choice.message.content.strip()

        # extract logprobs for d, r, u, l if available
        import math
        prob_vector = [0.0, 0.0, 0.0, 0.0]  # d, r, u, l
        target_tokens = ['d', 'r', 'u', 'l']
        if choice.logprobs and choice.logprobs.content:
            top_logprobs = choice.logprobs.content[0].top_logprobs
            for entry in top_logprobs:
                token = entry.token.strip().lower()
                if token in target_tokens:
                    idx = target_tokens.index(token)
                    prob_vector[idx] += math.exp(entry.logprob)

        return argmax_token, prob_vector


def api_converse(config, client, starting_prompts):
    # initialize agents
    agents = {
        agent_id: Agent(config, 0, agent_id, client)
        for agent_id in range(config['num_agents'])
    }
    # conversation loop
    prompt = ""
    for r in range(config['num_rounds']):
        print(f"ROUND {r}:")
        agent_id = r % config['num_agents']  # determine which agent acts this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(f"PROMPT: {prompt}")
        argmax_token, prob_vector = agent.generate_response(prompt)
        rounded_prob_vector = [round(p, 4) for p in prob_vector]
        print(f"Round {r}: agent {agent_id} generated response: {argmax_token} with probabilities for [d,r,u,l]: {rounded_prob_vector}")

        # update prompt for next round
        prompt = f"The other agent said: {argmax_token} with probabilities for [d,r,u,l]: {rounded_prob_vector}. Respond with one of [d,r,u,l]."

    return argmax_token

if __name__ == "__main__":
    # load API key
    api_key = Path("gpt_api_key.txt").read_text().strip()
    client = OpenAI(api_key=api_key)

    config = {
        "model_name": "gpt-4o",  # model to use for the conversation
        "max_new_tokens": 5,
        "temperature": 0.7,
        "top_p": 0.01,
        "num_rounds": 4,
        "num_agents": 2,
    }

    maze_str0 = "@#??#??#??.??.??.?.?.##?#..##?.#.??*"
    maze_str1 = "@?#.?..?#.?..?.#?.?#???.?????#???..*"

    def generate_starting_prompt(maze_str, prefix):
        formatted_maze = format_maze(maze_str)
        return f"""Task: You are going to play the collaborative maze game together with another agent. The maze consists of a grid with walls and a goal. You will receive a prefix of the path to the goal. Your task is to determine the next move on the path. You will each get your own map of the same maze, with some coordinates hidden. Because of the hidden coordinates, you will need to communicate with the other agent to share information about the maze and coordinate your next move.
            Rules:
            - You can only move to adjacent cells (up, down, left, right).
            - You can not move diagonally or through walls.
            Here is **your** map of the maze with a legend of the symbols:
            {formatted_maze}
            Legend:
            @ - Current Position
            * - Goal Position
            . - Path
            # - Wall
            ? - Hidden Cell
            Here is the path to the goal that we have moved so far: {prefix}. You should extract your current position from this.
            Then, respond with one of [d, r, u, l]. Do not include any other text.
            """

    prefix = ""
    label_sequence = "ddrrdddrrr"

    for i in range(len(label_sequence)):
        starting_prompts = [generate_starting_prompt(maze_str0, prefix), generate_starting_prompt(maze_str1, prefix)]    
        argmax_token = api_converse(config, client, starting_prompts)
        prefix += argmax_token
        print(f"Updated prefix after round {i}: {prefix}")

    # Final evaluation of the generated path against the label sequence
    print(f"Final generated path: {prefix}")
    print(f"Label sequence: {label_sequence}")
    if prefix == label_sequence:
        print("Success! The generated path matches the label sequence.")
    else:
        print("The generated path does not match the label sequence.")
