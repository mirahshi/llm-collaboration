#!/usr/bin/env python3
"""
Simple local inference script using Hugging Face transformers.
Edit MODEL_NAME and INPUT_TEXT below, then run:
    python pretrain_converse.py
"""

from __future__ import annotations
from termcolor import colored
from tqdm import tqdm

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
        input_ids = model_inputs["input_ids"]

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

        # with torch.no_grad():
        #     outputs = self.model(**model_inputs)
        # argmax_id = torch.argmax(outputs.logits[0, -1, :], dim=-1)
        # argmax_token = self.tokenizer.decode([[argmax_id.item()]])
        # probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        # prob_vector = [probs[i].item() for i in [67, 81, 84, 75]] # d, r, u, l
        output_ids, prob_vector = self.generate(input_ids, self.config['max_new_tokens'])
        generated_ids = output_ids[0][input_ids.shape[-1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        return response, prob_vector

    def format_inputs(self, prompt: str):
        """
        Text tokenization.
        """
        # user_prompt = "/no_think" + prompt  # turn off thinking
        user_prompt = prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(chat_prompt, return_tensors="pt")

        # Fallback for non-chat tokenizers.
        plain_prompt = f"User: {user_prompt}\nAssistant:"
        return self.tokenizer(plain_prompt, return_tensors="pt")
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, delimiter='~'):
        stop_generation = False
        for i in tqdm(range(max_new_tokens)):
            outputs = self.model(idx)
            argmax_id = torch.argmax(outputs.logits[0, -1, :], dim=-1)
            argmax_token = self.tokenizer.decode([[argmax_id.item()]])
            idx = torch.cat((idx, argmax_id.reshape(1, 1)), dim=1) # append the argmax token to the input sequence

            if stop_generation:
                probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
                prob_vector = [probs[i].item() for i in [67, 81, 84, 75]] # d, r, u, l
                return idx, prob_vector
            if delimiter in argmax_token[0]:
                stop_generation = True
        return idx, None




def pretrain_converse(config, tokenizer, model, starting_prompts):
    # initialize agents
    agents = {
        agent_id: Agent(config, 0, agent_id, tokenizer, model)
        for agent_id in range(config['num_agents'])
    }
    # conversation loop
    prompt = ""
    for r in range(config['num_rounds']):
        print(colored(f"ROUND {r}: =================================================", 'light_yellow'))
        agent_id = r % config['num_agents'] # determine which agent trains this round
        agent = agents[agent_id]
        agent.round = r
        prompt = starting_prompts[agent_id] + prompt
        print(colored(f"PROMPT: {prompt}", 'light_blue'))
        response, prob_vector = agent.generate_response(prompt)
        if prob_vector is not None:
            prob_vector = [round(p, 2) for p in prob_vector]
        # argmax_token = argmax_token[0]
        print(f"Agent {agent_id} generated response: {response}")
        print(f"Probabilities for [d,r,u,l]: {prob_vector}")

        # update prompt for next round
        prompt = f"The other agent said: {response} with probabilities for [d,r,u,l]: {prob_vector}."

    return response

if __name__ == "__main__":
    config = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct", # hugging face model to load
    "model_path": "/vast/projects/surbhig/multi-agent-collab/pretrained_models", # path to cache the model
    "max_new_tokens": 512, # maximum number of new tokens to generate
    "temperature": 0.7, # temperature for sampling
    "top_p": 0.9, # top-p sampling
    "repetition_penalty": 1.1, # repetition penalty
    "num_rounds": 2, # number of rounds to train
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
    
    maze_str0 = "@##.#..##......#...#.##.#..###.#...*"
    maze_str1 = "@?#.?..?#.?..?.#?.?#???.?????#???..*"
    prefix = "ddrr"

    def generate_starting_prompt(maze_str, prefix):
        formatted_maze = format_maze(maze_str)
        return_string =  f"""Task: You are going to play the collaborative maze game together with another agent. The maze consists of a grid with walls and a goal. Your task is to determine the next move on the path. You will each get your own map of the same maze, with some coordinates hidden. Because of the hidden coordinates, you will need to communicate with the other agent to share information about the maze and coordinate your next move.
Rules:
- You can only move to adjacent cells [down(d), right(r), up(u), left(l)].
- You can not move diagonally or through walls.
Here is **your** map of the maze with a legend of the symbols: \n
{formatted_maze} \n
Legend:
@ - Current Position
* - Goal Position
. - Path
# - Wall
? - Hidden Cell
"""
        if prefix != "":
            return_string += f"""
This is the path to the goal that we have moved so far from which you can infer your current position: {prefix}
"""
        return_string += """
Explain your reasoning, then answer with one of d, r, u, l. Put your final answer after the delimiter ~. For example, if your final answer is d, your response should contain ~d. 
"""
        return return_string

    # starting_prompts = [generate_starting_prompt(maze_str0, prefix), generate_starting_prompt(maze_str1, prefix)]
    # pretrain_converse(config, tokenizer, model, starting_prompts)

    prefix = ""
    label_sequence = "ddrrdddrrr"

    # for i in range(len(label_sequence)):
    for i in range(1):
        starting_prompts = [generate_starting_prompt(maze_str0, prefix), generate_starting_prompt(maze_str1, prefix)]    
        response = pretrain_converse(config, tokenizer, model, starting_prompts)
        prefix += response[-1]
        print(f"Updated prefix after move {i}: {prefix}")

    # Final evaluation of the generated path against the label sequence
    print(f"Final generated path: {prefix}")
    print(f"Label sequence: {label_sequence}")
    if prefix == label_sequence:
        print("Success! The generated path matches the label sequence.")
    else:
        print("The generated path does not match the label sequence.")