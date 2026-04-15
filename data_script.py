import os
import numpy as np

# get maze conversation logs with probabiliites from round 0 and round 1 for training calibrator starting from round 1
maze_conversation_log_path = "out-api_exp3/probs-rollouts/conversations"
save_path = "out-api_exp4/probs-rollouts/conversations"
os.makedirs(save_path, exist_ok=True)

for i in range(100):
    if os.path.exists(os.path.join(maze_conversation_log_path, f"maze_{i}.npy")):
        maze_conversation_logs = np.load(os.path.join(maze_conversation_log_path, f"maze_{i}.npy"), allow_pickle=True).item()
        prior_conversation_logs = {i: []}
        for j, conversation_log in enumerate(maze_conversation_logs[i]):
            prior_conversation_log = {
                'full_responses': conversation_log['full_responses'][0:2],
                'final_answers': conversation_log['final_answers'][0:2],
                'prob_vectors': conversation_log['prob_vectors'][0:2],
                'calibrated_prob_vectors': [None], 
                'format_failures': conversation_log['format_failures'][0:2],
                'label': conversation_log['label'],
            }
            prior_conversation_logs[i].append(prior_conversation_log)
        np.save(os.path.join(save_path, f"maze_{i}.npy"), prior_conversation_logs)

for i in range(100):
    if os.path.exists(os.path.join(save_path, f"maze_{i}.npy")):
        prior_conversation_logs = np.load(os.path.join(save_path, f"maze_{i}.npy"), allow_pickle=True).item()
        print(prior_conversation_logs)
        exit()


