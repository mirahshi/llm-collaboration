import numpy as np
import os
from termcolor import colored
from glob import glob

def get_maze_conversation_logs(maze_conversation_log_path):
    maze_conversation_logs = {}
    maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
    for maze_path in maze_paths:
        maze_conversation_logs_loaded = np.load(maze_path, allow_pickle=True).item()
        # append contents of maze_conversation_logs_loaded to maze_conversation_logs
        maze_conversation_logs.update(maze_conversation_logs_loaded)
    print("maze_conversation_logs: ", len(maze_conversation_logs))
    return maze_conversation_logs

def get_maze_conversation_logs_no_format_failures(maze_conversation_logs):
    """
    Filter maze_conversation_logs to only include mazes with no format failures.
    
    Args:
        maze_conversation_logs: dict (or numpy array containing dict) mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict with same structure, but only mazes where no move had any format failure
    """
    # Handle numpy array from np.load
    if isinstance(maze_conversation_logs, np.ndarray):
        maze_conversation_logs = maze_conversation_logs.item()
    
    maze_conversation_logs_no_format_failures = {}
    for i, conversation_logs in maze_conversation_logs.items():
        has_format_failure = False
        for conversation_log in conversation_logs:
            if any(conversation_log['format_failures']):
                has_format_failure = True
                break
        if not has_format_failure:
            maze_conversation_logs_no_format_failures[i] = conversation_logs
    return maze_conversation_logs_no_format_failures

def get_label_sequences(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    label_sequences = [line.split('=')[1].strip() for line in lines]
    return label_sequences

def measure_zero_one_losses(maze_conversation_logs, label_sequences):
    """
    Compute average 0/1 loss per round.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
        label_sequences: list of label sequences (strings), one per maze
    
    Returns:
        dict mapping round number to average 0/1 loss
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    losses = {}
    for r in range(num_rounds):
        round_losses = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                final_answer = conversation_log['final_answers'][r]
                label = label_sequences[i][j]
                loss = 0 if final_answer == label else 1
                round_losses.append(loss)
        losses[r] = np.mean(round_losses)
    return losses

def measure_disagreement(maze_conversation_logs):
    """
    Compute average L2 distance between prob vectors of consecutive rounds.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict mapping round number r to average L2 distance between prob vectors at r-1 and r
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    distances = {}
    for r in range(1, num_rounds):
        round_distances = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                prob_prev = conversation_log['prob_vectors'][r - 1]
                prob_curr = conversation_log['prob_vectors'][r]
                if prob_prev is not None and prob_curr is not None:
                    l2_dist = np.linalg.norm(np.array(prob_prev) - np.array(prob_curr))
                    round_distances.append(l2_dist)
        if round_distances:
            distances[r] = np.mean(round_distances)
        else:
            distances[r] = None
    return distances


def measure_maze_success_rates(maze_conversation_logs, label_sequences):
    """
    Compute maze success rate per round.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
        label_sequences: list of label sequences (strings), one per maze
    
    Returns:
        dict mapping round number to maze success rate (fraction of mazes with all moves correct)
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    success_rates = {}
    for r in range(num_rounds):
        successful_mazes = 0
        total_mazes = len(maze_conversation_logs)
        for i, conversation_logs in maze_conversation_logs.items():
            maze_correct = True
            for j, conversation_log in enumerate(conversation_logs):
                final_answer = conversation_log['final_answers'][r]
                label = label_sequences[i][j]
                if final_answer != label:
                    maze_correct = False
                    break
            if maze_correct:
                successful_mazes += 1
        success_rates[r] = successful_mazes / total_mazes
    return success_rates


if __name__ == "__main__":
    data_dir = 'out-api_exp1'
    group_name = "api_exp3"
    run_names = ["probs-rollouts"]
    # run_names = ["probs-verbalized-gpt-4.1"]

    # Load and filter all conditions first
    all_logs = {}
    all_filtered_logs = {}
    for run_name in run_names:
        out_dir = f'out-{group_name}/{run_name}'
        maze_conversation_logs_path = os.path.join(out_dir, 'conversations')
        maze_conversation_logs = get_maze_conversation_logs(maze_conversation_logs_path)
        # maze_conversation_logs = np.load(os.path.join(out_dir, 'maze_conversation_logs.npy'), allow_pickle=True).item()
        all_logs[run_name] = maze_conversation_logs
        all_filtered_logs[run_name] = get_maze_conversation_logs_no_format_failures(maze_conversation_logs)
        print(f"{run_name}: {len(all_filtered_logs[run_name])} mazes with no format failures")

    # Find intersection of maze indices with no format failures in all conditions
    common_maze_indices = set(all_filtered_logs[run_names[0]].keys())
    for run_name in run_names[1:]:
        common_maze_indices &= set(all_filtered_logs[run_name].keys())
    print(colored(f"Common mazes with no format failures in all conditions: {len(common_maze_indices)}", 'light_green'))

    label_sequences = get_label_sequences(os.path.join(data_dir, 'input0.txt'))

    # Evaluate each condition on the common set of mazes
    for run_name in run_names:
        print(colored(f"\nEvaluating {run_name} (on {len(common_maze_indices)} common mazes)", 'light_yellow'))
        
        # Filter to only common mazes
        filtered_logs = {i: all_filtered_logs[run_name][i] for i in common_maze_indices}
        
        losses = measure_zero_one_losses(filtered_logs, label_sequences)
        success_rates = measure_maze_success_rates(filtered_logs, label_sequences)
        disagreements = measure_disagreement(filtered_logs)
        print(colored("zero-one losses:", 'light_blue'))
        for r, loss in losses.items():
            print(f"Round {r}: {loss}")
        print(colored("maze success rates:", 'light_blue'))
        for r, success_rate in success_rates.items():
            print(f"Round {r}: {success_rate}")
        print(colored("disagreement:", 'light_blue'))
        for r, disagreement in disagreements.items():
            print(f"Round {r}: {disagreement}")