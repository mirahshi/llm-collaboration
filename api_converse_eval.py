import numpy as np
import os
from termcolor import colored
from glob import glob
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
    Filter maze_conversation_logs to only include mazes with no format failures. And remove mazes with no conversation logs.
    
    Args:
        maze_conversation_logs: dict (or numpy array containing dict) mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict with same structure, but only mazes where no move had any format failure
    """
    # Handle numpy array from np.load
    if isinstance(maze_conversation_logs, np.ndarray):
        maze_conversation_logs = maze_conversation_logs.item()
    
    maze_conversation_logs_no_format_failures = {}
    maze_indices_with_format_failure = []
    for i, conversation_logs in maze_conversation_logs.items():
        has_format_failure = False
        for conversation_log in conversation_logs:
            if any(conversation_log['format_failures']):
                maze_indices_with_format_failure.append(i)
                has_format_failure = True
                break
        if not has_format_failure and len(conversation_logs) > 0:
            maze_conversation_logs_no_format_failures[i] = conversation_logs
    print(f"Maze indices with format failure: {maze_indices_with_format_failure}")
    return maze_conversation_logs_no_format_failures

def _ci95(values):
    """Return 1.96 * standard error of the mean for a list/array of values."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return 0.0
    return 1.96 * arr.std(ddof=1) / np.sqrt(n)

def measure_zero_one_losses(maze_conversation_logs, num_rounds):
    """
    Compute average 0/1 loss per round.

    For each (maze, move, round), the predicted answer is:
      - argmax of calibrated_prob_vectors[r] if available, else
      - argmax of prob_vectors[r] (falling back from final_answers[r] if it disagrees)

    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)

    Returns:
        losses: dict mapping round number to average 0/1 loss
        losses_ci: dict mapping round number to 1.96 * standard error of the mean
        non_argmax_count: int, number of times final_answers[r] disagreed with argmax(prob_vectors[r])
    """
    answer_tokens = ['d', 'r', 'u', 'l']
    answer_tokens = ['d', 'r', 'u', 'l']
    first_maze_idx = next(iter(maze_conversation_logs))

    losses = {}
    losses_ci = {}
    non_argmax_count = 0
    for r in range(num_rounds):
        round_losses = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                calibrated = conversation_log.get('calibrated_prob_vectors', [None] * num_rounds)
                if calibrated[r] is not None:
                    final_answer = answer_tokens[np.argmax(calibrated[r])]
                    # final_answer = conversation_log['final_answers'][r] # use raw final answer from model
                else:
                    final_answer = conversation_log['final_answers'][r]
                    argmax_answer = answer_tokens[np.argmax(conversation_log['prob_vectors'][r])]
                    if final_answer != argmax_answer:
                        non_argmax_count += 1
                        final_answer = argmax_answer
                label = conversation_log['label']
                loss = 0 if final_answer == label else 1
                round_losses.append(loss)
        losses[r] = np.mean(round_losses)
        losses_ci[r] = _ci95(round_losses)
    return losses, losses_ci, non_argmax_count

def measure_disagreement(maze_conversation_logs, num_rounds):
    """
    Compute average L2 distance between prob vectors of consecutive rounds.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        distances: dict mapping round number r to average L2 distance between prob vectors at r-1 and r
        distances_ci: dict mapping round number r to 1.96 * standard error of the mean
    """

    distances = {}
    distances_ci = {}
    for r in range(1, num_rounds):
        round_distances = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                if conversation_log['calibrated_prob_vectors'][r - 1] is not None: # use calibrated probabilities from previous round
                    prob_prev = conversation_log['calibrated_prob_vectors'][r - 1]
                else: # use raw probabilities from previous round
                    prob_prev = conversation_log['prob_vectors'][r - 1]
                if conversation_log['calibrated_prob_vectors'][r] is not None: # use calibrated probabilities from current round
                    prob_curr = conversation_log['calibrated_prob_vectors'][r]
                else: # use raw probabilities from current round
                    prob_curr = conversation_log['prob_vectors'][r]
                if prob_prev is not None and prob_curr is not None:
                    l2_dist = np.linalg.norm(np.array(prob_prev) - np.array(prob_curr))
                    round_distances.append(l2_dist)
        if round_distances:
            distances[r] = np.mean(round_distances)
            distances_ci[r] = _ci95(round_distances)
        else:
            distances[r] = None
            distances_ci[r] = None
    return distances, distances_ci


def measure_maze_success_rates(maze_conversation_logs, num_rounds):
    """
    Compute maze success rate per round.

    Uses the same calibrated-aware argmax answer-selection rule as measure_zero_one_losses.

    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)

    Returns:
        success_rates: dict mapping round number to maze success rate (fraction of mazes with all moves correct)
        success_rates_ci: dict mapping round number to 1.96 * standard error of the mean
    """
    answer_tokens = ['d', 'r', 'u', 'l']
    first_maze_idx = next(iter(maze_conversation_logs))

    success_rates = {}
    success_rates_ci = {}
    for r in range(num_rounds):
        per_maze_success = []
        for i, conversation_logs in maze_conversation_logs.items():
            maze_correct = True
            for j, conversation_log in enumerate(conversation_logs):
                calibrated = conversation_log.get('calibrated_prob_vectors', [None] * num_rounds)
                if calibrated[r] is not None:
                    final_answer = answer_tokens[np.argmax(calibrated[r])]
                    # final_answer = conversation_log['final_answers'][r] # use raw final answer from model
                else:
                    final_answer = conversation_log['final_answers'][r]
                    argmax_answer = answer_tokens[np.argmax(conversation_log['prob_vectors'][r])]
                    if final_answer != argmax_answer:
                        final_answer = argmax_answer
                label = conversation_log['label']
                if final_answer != label:
                    maze_correct = False
                    break
            per_maze_success.append(1 if maze_correct else 0)
        success_rates[r] = np.mean(per_maze_success)
        success_rates_ci[r] = _ci95(per_maze_success)
    return success_rates, success_rates_ci

def generate_calibration_data(maze_conversation_logs, round):
    """
    Generate calibration data for a given round of the maze conversation logs.
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
        round > 0: int, the round to generate calibration data for
    Returns:
        probs, collaborator_probs: list of probability vectors
        labels: token indices of the correct answer tokens
    """
    probs = []
    collaborator_probs = []
    labels = []

    answer_to_indices = {'d': 0, 'r': 1, 'u': 2, 'l': 3}

    for i, conversation_logs in maze_conversation_logs.items():
        for j, conversation_log in enumerate(conversation_logs):
            # if round == 3:
            #     print("raw prob: ", conversation_log['prob_vectors'][round])
            if conversation_log['calibrated_prob_vectors'][round] is not None: # use calibrated probabilities from current round
                prob = conversation_log['calibrated_prob_vectors'][round]
                # if round == 3:
                #     print("calibrated prob: ", prob)
            else: # use raw probabilities from current round
                prob = conversation_log['prob_vectors'][round]
            
            if round > 1 and conversation_log['calibrated_prob_vectors'][round - 1] is not None: # for round 2 and beyond, use calibrated probabilities from previous round
                collaborator_prob = conversation_log['calibrated_prob_vectors'][round - 1]
            elif round == 0:
                collaborator_prob = None
            else: # for round 1, use raw probabilities from round 0
                collaborator_prob = conversation_log['prob_vectors'][round - 1]
            # if round > 0:
            #     collaborator_prob = conversation_log['prob_vectors'][round - 1]
            # else:
            #     collaborator_prob = None
            label_token = conversation_log['label']
            # print(f"collaborator_prob: {collaborator_prob}, prob: {prob}, label: {label_token}")

            probs.append(prob)
            collaborator_probs.append(collaborator_prob)
            labels.append(answer_to_indices[label_token])
    
    if round == 0:
        collaborator_probs = None
    else:
        collaborator_probs = torch.tensor(collaborator_probs, dtype=torch.float32)
                
    return torch.tensor(probs, dtype=torch.float32), collaborator_probs, torch.tensor(labels, dtype=torch.int64)

def ECE_multidim_new(probs, labels, K=5, K_cross=5, smooth=False, sigma=0.1, collaborator_probs=None, confidence=True):
    """
    Computes ECE loss on probabilities and labels. 
    If smooth is False, uses ECE loss (K buckets).
    If smooth is True, uses smooth ECE loss (Gaussian weights centered at K points with std dev sigma).
    if collaborator_probs is not None, use it to compute cross ECE.
    """

    if confidence:
        # sample predictions from probs
        sampled_predictions = torch.distributions.Categorical(probs=probs).sample()
        conf_labels = torch.gather(labels, dim=-1, index=sampled_predictions.unsqueeze(-1))
        conf_labels = conf_labels.squeeze(-1)
        labels = conf_labels.unsqueeze(-1) # shape B x T x 1 (D = 1)
        # use max probabilities as predictions
        predictions = torch.max(probs, dim=-1).values.unsqueeze(-1) # shape B x T x 1 (D = 1)
        # TODO: specify collaborator predictions
    else:
        predictions = probs
        if collaborator_probs is not None:
            collaborator_predictions = collaborator_probs
    
    B, D = predictions.shape # prediction dimension
    N = B # number of examples
    num_buckets = K ** D # number of buckets

    flat_predictions = predictions.reshape(N, D)
    flat_labels = labels.to(dtype=probs.dtype).reshape(N, D)
            
    if smooth: # memory intensive
        # soft bucket assignment: define D-dimensional soft buckets using Gaussian kernels
        centers_1d = torch.linspace(0, 1, K, device=probs.device, dtype=probs.dtype)
        grids = torch.meshgrid([centers_1d] * D, indexing="ij")
        centers = torch.stack(grids, dim=-1).reshape(-1, D)  # shape K^D x D
        diff = predictions.unsqueeze(-2) - centers.unsqueeze(0)  # shape B x K^D x D
        diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x K^D; L2 norm squared across D
        unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
        weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # shape B x K^D

        if collaborator_probs is not None: # cross ECE buckets are joint buckets of the model and the collaborator
            # define D-dimensional soft buckets using Gaussian kernels
            centers_1d_collab = torch.linspace(0, 1, K_cross, device=probs.device, dtype=probs.dtype)
            grids_collab = torch.meshgrid([centers_1d_collab] * D, indexing="ij")
            centers_collab = torch.stack(grids_collab, dim=-1).reshape(-1, D)  # shape K_cross^D x D
            diff_collab = collaborator_predictions.unsqueeze(-2) - centers_collab.unsqueeze(0)  # shape B x K_cross^D x D
            diff_norm_collab = (torch.pow(diff_collab, 2)).sum(dim=-1)  # shape B x K_cross^D; L2 norm across D
            unnorm_weights_collab = torch.exp(- (diff_norm_collab) / (2 * sigma ** 2))
            weights_collaborator = unnorm_weights_collab / unnorm_weights_collab.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # shape B x K_cross^D

            joint_weights = weights_collaborator.unsqueeze(-1) * weights.unsqueeze(-2)  # shape B x K_cross^D x K^D
            joint_weights = joint_weights.reshape(N, -1)  # flatten to shape B x (K_cross^D * K^D)
            weights = joint_weights

        S = weights.sum(dim=0)  # sum of weights for each bucket

        p_bar = (weights.unsqueeze(-1) * predictions.unsqueeze(-2)).sum(dim=0) / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = (weights.unsqueeze(-1) * labels.unsqueeze(-2)).sum(dim=0) / S.clamp_min(1e-12).unsqueeze(-1)
    else: # memory efficient version
        # hard bucket assignment: each example gets one D-dimensional bucket
        edges_1d = torch.linspace(0, 1, K + 1, device=probs.device, dtype=probs.dtype)
        idx_per_coord = torch.bucketize(flat_predictions, edges_1d, right=True) - 1  # N x D: index of 1-dim bucket that prediction[n,d] is in
        idx_per_coord.clamp_(min=0, max=K - 1)
        powers = (K ** torch.arange(D, device=probs.device, dtype=torch.int64))  # D
        own_idx = (idx_per_coord.to(torch.int64) * powers).sum(dim=-1)  # N: index of D-dim bucket that prediction[n] is in
        if collaborator_probs is None:
            # use active buckets only
            active_bucket_ids, inverse = torch.unique(own_idx, sorted=False, return_inverse=True)
            M = active_bucket_ids.numel()  # M <= N

            S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
            wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
            wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

            ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
            S.scatter_add_(0, inverse, ones) # number of examples in each bucket
            inverse_exp = inverse.unsqueeze(-1).expand(-1, D) 
            wp.scatter_add_(0, inverse_exp, flat_predictions) # weighted sum of predictions in each bucket
            wl.scatter_add_(0, inverse_exp, flat_labels) # weighted sum of labels in each bucket
        else:
            flat_collab_predictions = collaborator_predictions.reshape(N, D)
            edges_1d_collab = torch.linspace(0, 1, K_cross + 1, device=probs.device, dtype=probs.dtype)
            collab_coord_idx = torch.bucketize(flat_collab_predictions, edges_1d_collab, right=True) - 1
            collab_coord_idx.clamp_(0, K_cross - 1)
            powers_collab = (K_cross ** torch.arange(D, device=probs.device, dtype=torch.int64))
            collab_idx = (collab_coord_idx.to(torch.int64) * powers_collab).sum(dim=-1)

            joint_idx_raw = own_idx + (K ** D) * collab_idx
            _, joint_inverse = torch.unique(joint_idx_raw, sorted=False, return_inverse=True) # index of joint bucket that prediction[n] is in

            M = int(joint_inverse.max().item()) + 1 # number of joint buckets

            S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
            wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
            wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

            ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
            S.scatter_add_(0, joint_inverse, ones)
            joint_exp = joint_inverse.unsqueeze(-1).expand(-1, D)
            wp.scatter_add_(0, joint_exp, flat_predictions)
            wl.scatter_add_(0, joint_exp, flat_labels)

        p_bar = wp / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = wl / S.clamp_min(1e-12).unsqueeze(-1)
        
    ece_loss = (S * torch.linalg.norm(p_bar - y_bar, ord=2, dim=-1)).sum() / N

    return ece_loss

def measure_calibration_losses(maze_conversation_logs, num_rounds, n_bootstrap=200, bootstrap_seed=0):
    """
    Compute the calibration losses for each round, with bootstrap-based 1.96 SE estimates.
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
        num_rounds: number of rounds to evaluate
        n_bootstrap: number of bootstrap resamples used to estimate the standard error
        bootstrap_seed: RNG seed for reproducible resampling
    Returns:
        ece_losses: dict mapping round number to calibration loss
        ece_losses_ci: dict mapping round number to 1.96 * bootstrap standard error
        cross_ece_losses: dict mapping round number to cross-calibration loss
        cross_ece_losses_ci: dict mapping round number to 1.96 * bootstrap standard error
    """
    answer_tokens = ['d', 'r', 'u', 'l']

    ece_losses = {}
    ece_losses_ci = {}
    cross_ece_losses = {}
    cross_ece_losses_ci = {}
    rng = np.random.default_rng(bootstrap_seed)
    for r in range(num_rounds):
        probs, collaborator_probs, labels = generate_calibration_data(maze_conversation_logs, r)
        labels_one_hot = F.one_hot(labels, num_classes=len(answer_tokens))
        ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
        ece_losses[r] = ece_loss.item()

        n = probs.shape[0]
        boot_ece = np.empty(n_bootstrap, dtype=np.float64)
        boot_cross = np.empty(n_bootstrap, dtype=np.float64) if r > 0 else None
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            idx_t = torch.from_numpy(idx).long()
            probs_b = probs[idx_t]
            labels_b = labels_one_hot[idx_t]
            boot_ece[b] = ECE_multidim_new(probs_b, labels_b, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False).item()
            if r > 0:
                collab_b = collaborator_probs[idx_t]
                boot_cross[b] = ECE_multidim_new(probs_b, labels_b, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collab_b, confidence=False).item()
        ece_losses_ci[r] = 1.96 * boot_ece.std(ddof=1) if n_bootstrap > 1 else 0.0

        if r > 0:
            cross_ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=False)
            cross_ece_losses[r] = cross_ece_loss.item()
            cross_ece_losses_ci[r] = 1.96 * boot_cross.std(ddof=1) if n_bootstrap > 1 else 0.0
    return ece_losses, ece_losses_ci, cross_ece_losses, cross_ece_losses_ci

def plot_losses_and_success_rates(results_per_run, output_path, title=None):
    """
    Plot a grouped bar graph of 0/1 losses and maze success rates over rounds for each run.

    Args:
        results_per_run: dict mapping run_name to {'losses': {r: val}, 'success_rates': {r: val}}
        output_path: path to save the figure
    """
    run_names = list(results_per_run.keys())
    rounds = sorted(next(iter(results_per_run.values()))['losses'].keys())
    n_rounds = len(rounds)
    n_runs = len(run_names)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.5 * n_rounds), 5))
    metric_keys = [('losses', '0/1 loss'), ('success_rates', 'maze success rate')]

    bar_width = 0.8 / max(n_runs, 1)
    x = np.arange(n_rounds)

    max_loss = max(
        (max([results_per_run[run_name]['losses'][r] for r in rounds]) if results_per_run[run_name]['losses'] else 0)
        for run_name in run_names
    )
    max_success = max(
        (max([results_per_run[run_name]['success_rates'][r] for r in rounds]) if results_per_run[run_name]['success_rates'] else 0)
        for run_name in run_names
    )
    max_y = max(max_loss, max_success)
       

    for ax, (key, ylabel) in zip(axes, metric_keys):
        ci_key = f'{key}_ci'
        for k, run_name in enumerate(run_names):
            vals = [results_per_run[run_name][key][r] for r in rounds]
            ci_dict = results_per_run[run_name].get(ci_key, {})
            errs = [ci_dict.get(r, 0.0) or 0.0 for r in rounds]
            offsets = x + (k - (n_runs - 1) / 2) * bar_width
            bars = ax.bar(offsets, vals, bar_width, label=run_name, yerr=errs, capsize=3)
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Round {r}' for r in rounds])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' over rounds')
        ax.set_ylim(0, max_y + 0.1)
   
        if n_runs > 1:
            ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(colored(f"Saved bar graph: {output_path}", 'light_green'))


if __name__ == "__main__":
    group_name = "api_exp10"
    run_names = ["verbalized-calibrator-self"]
    conversations_subdirs = ["conversations3"]
    maze_conversation_logs_paths = [
        f'/vast/projects/surbhig/multi-agent-collab/out-{group_name}/{run_name}/{subdir}'
        for run_name, subdir in zip(run_names, conversations_subdirs)
    ]
    num_rounds = 3
    filter_common_mazes = False  # if True, evaluate each run only on the intersection of mazes with no format failures across all runs
    plot = False

    # Load and filter all conditions first
    all_logs = {}
    all_filtered_logs = {}
    for i, run_name in enumerate(run_names):
        maze_conversation_logs_path = maze_conversation_logs_paths[i]
        maze_conversation_logs = get_maze_conversation_logs(maze_conversation_logs_path)
        all_logs[run_name] = maze_conversation_logs
        all_filtered_logs[run_name] = get_maze_conversation_logs_no_format_failures(maze_conversation_logs)
        print(f"{run_name}: {len(all_filtered_logs[run_name])} mazes with no format failures in {maze_conversation_logs_path}")

    if filter_common_mazes:
        # Find intersection of maze indices with no format failures in all conditions
        common_maze_indices = set(all_filtered_logs[run_names[0]].keys())
        for run_name in run_names[1:]:
            common_maze_indices &= set(all_filtered_logs[run_name].keys())
        print(colored(f"Common mazes with no format failures in all conditions: {len(common_maze_indices)}", 'light_green'))
        print(colored(f"Total number of examples in common mazes with no format failures: {sum([len(all_filtered_logs[run_name][i]) for i in common_maze_indices])}", 'light_green'))
    else:
        common_maze_indices = None
        print(colored("Common-maze filtering disabled — evaluating each run on all its no-format-failure mazes", 'light_green'))

    # Evaluate each condition on the (optionally common) set of mazes
    results_per_run = {}
    for run_name in run_names:
        if filter_common_mazes:
            filtered_logs = {i: all_filtered_logs[run_name][i] for i in common_maze_indices}
            print(colored(f"\nEvaluating {run_name} (on {len(common_maze_indices)} common mazes)", 'light_yellow'))
        else:
            filtered_logs = all_filtered_logs[run_name]
            print(colored(f"\nEvaluating {run_name} (on {len(filtered_logs)} mazes)", 'light_yellow'))

        losses, losses_ci, _non_argmax_count = measure_zero_one_losses(filtered_logs, num_rounds)
        success_rates, success_rates_ci = measure_maze_success_rates(filtered_logs, num_rounds)
        disagreements, disagreements_ci = measure_disagreement(filtered_logs, num_rounds)
        ece_losses, ece_losses_ci, cross_ece_losses, cross_ece_losses_ci = measure_calibration_losses(filtered_logs, num_rounds)

        results_per_run[run_name] = {
            'losses': losses, 'losses_ci': losses_ci,
            'success_rates': success_rates, 'success_rates_ci': success_rates_ci,
        }

        def _fmt(mean, ci):
            if mean is None:
                return "None"
            if ci is None:
                return f"{mean:.4f}"
            return f"{mean:.4f} ± {ci:.4f}"

        print(colored("zero-one losses:", 'light_blue'))
        for r, loss in losses.items():
            print(f"Round {r}: {_fmt(loss, losses_ci[r])}")
        print(colored("maze success rates:", 'light_blue'))
        for r, success_rate in success_rates.items():
            print(f"Round {r}: {_fmt(success_rate, success_rates_ci[r])}")
        print(colored("disagreement:", 'light_blue'))
        for r, disagreement in disagreements.items():
            print(f"Round {r}: {_fmt(disagreement, disagreements_ci[r])}")
        print(colored("ece losses:", 'light_blue'))
        for r, ece_loss in ece_losses.items():
            print(f"Round {r}: {_fmt(ece_loss, ece_losses_ci[r])}")
        print(colored("cross ece losses:", 'light_blue'))
        for r, cross_ece_loss in cross_ece_losses.items():
            print(f"Round {r}: {_fmt(cross_ece_loss, cross_ece_losses_ci[r])}")

    if plot:
        results_dir = f'/vast/projects/surbhig/multi-agent-collab/out-{group_name}/{run_names[0]}/results'
        os.makedirs(results_dir, exist_ok=True)
        runs_tag = "_".join(run_names)
        plot_path = os.path.join(results_dir, f'losses_success_rates_{runs_tag}.png')
        plot_losses_and_success_rates(results_per_run, plot_path, title=f'Baseline Verbalized Probs vs Calibrated Verbalized Probs')