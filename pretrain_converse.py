#!/usr/bin/env python3
"""
Simple local inference script using Hugging Face transformers.
Edit MODEL_NAME and INPUT_TEXT below, then run:
    python pretrain_converse.py
"""

from __future__ import annotations

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def format_maze(maze_str):
    maze_size = int(len(maze_str) ** 0.5)
    formatted_maze = "\n".join(maze_str[i:i+maze_size] for i in range(0, len(maze_str), maze_size))
    return formatted_maze

class Agent():
    def __init__(self, config, round, id, tokenizer, model):
        self.config = config
        self.round = round
        self.id = id
        self.tokenizer = tokenizer
        self.model = model

    def generate_response(self, prompt):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        model_inputs = self.format_inputs(prompt).to(self.model.device)
        # input_ids = model_inputs["input_ids"]

        # with torch.no_grad():
        #     output_ids = model.generate(
        #         **model_inputs,
        #         max_new_tokens=self.config['max_new_tokens'],
        #         do_sample=True,
        #         temperature=self.config['temperature'],
        #         top_p=self.config['top_p'],
        #         repetition_penalty=self.config['repetition_penalty'],
        #         pad_token_id=tokenizer.pad_token_id,
        #         eos_token_id=tokenizer.eos_token_id,
        #     )

        # generated_ids = output_ids[0][input_ids.shape[-1] :]
        # response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # print("\n=== Model Response ===")
        # print(response)

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        argmax_id = torch.argmax(outputs.logits[0, -1, :], dim=-1)
        argmax_token = self.tokenizer.decode([[argmax_id.item()]])
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        prob_vector = [probs[i].item() for i in [67, 81, 84, 75]] # d, r, u, l

        return argmax_token, prob_vector

    def format_inputs(self, prompt: str):
        """
        Text tokenization.
        """
        prompt = "/no_think" + prompt # turn off thinking
        return self.tokenizer(prompt, return_tensors="pt")

def pretrain_converse(config, tokenizer, model, starting_prompts):
    # initialize agents
    agents = {
        agent_id: Agent(config, 0, agent_id, tokenizer, model)
        for agent_id in range(config['num_agents'])
    }
    # conversation loop
    prompt = ""
    for r in range(config['num_rounds']):
        print(f"ROUND {r}:")
        agent_id = r % config['num_agents'] # determine which agent trains this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(f"PROMPT: {prompt}")
        argmax_token, prob_vector = agent.generate_response(prompt)
        rounded_prob_vector = [round(p, 2) for p in prob_vector]
        print(f"Round {r}: agent {agent_id} generated response: {argmax_token} with probabilities for [d,r,u,l]: {rounded_prob_vector}")

        # update prompt for next round
        prompt = f"The other agent said: {argmax_token} with probabilities for [d,r,u,l]: {rounded_prob_vector}. Respond with one of [d,r,u,l]."


if __name__ == "__main__":
    config = {
    "model_name": "Qwen/Qwen3.5-2B", # hugging face model to load
    "model_path": "/home/mirahshi/projects/llm-collaboration/pretrained_models", # path to cache the model
    "max_new_tokens": 5, # maximum number of new tokens to generate
    "temperature": 0.7, # temperature for sampling
    "top_p": 0.9, # top-p sampling
    "repetition_penalty": 1.1, # repetition penalty
    "num_rounds": 4, # number of rounds to train
    "num_agents": 2, # number of agents
    }
    
    # preload tokenizer and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_source = config['model_name']
    cache_dir = config['model_path'].strip() or None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model on {device} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    maze_str0 = "@#??#??#??.??.??.?.?.##?#..##?.#.??*"
    maze_str1 = "@?#.?..?#.?..?.#?.?#???.?????#???..*"
    prefix = "ddrr"

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
Here is the prefix of the path to the goal: {prefix}
Respond with one of [d, r, u, l]. Do not include any other text.
"""

    starting_prompts = [generate_starting_prompt(maze_str0, prefix), generate_starting_prompt(maze_str1, prefix)]
    pretrain_converse(config, tokenizer, model, starting_prompts)