import os
import numpy as np
from glob import glob
from tqdm import tqdm

"""
Get maze conversation logs with probabiliites from round 0 and round 1 for training calibrator starting from round 1
"""

maze_conversation_log_path = "/vast/projects/surbhig/multi-agent-collab/out-api_exp10/verbalized/conversations"
save_path = "/vast/projects/surbhig/multi-agent-collab/out-api_exp10/verbalized-calibrator/conversations1"
os.makedirs(save_path, exist_ok=True)

maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
for maze_path in tqdm(maze_paths):
    maze_conversation_logs = np.load(maze_path, allow_pickle=True).item()
    assert len(maze_conversation_logs) == 1, "Expected only one maze in the conversation logs"
    i = list(maze_conversation_logs.keys())[0] # get maze index

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

# for i in range(100):
#     if os.path.exists(os.path.join(maze_conversation_log_path, f"maze_{i}.npy")):
#         maze_conversation_logs = np.load(os.path.join(maze_conversation_log_path, f"maze_{i}.npy"), allow_pickle=True).item()
#         prior_conversation_logs = {i: []}
#         for j, conversation_log in enumerate(maze_conversation_logs[i]):
#             prior_conversation_log = {
#                 'full_responses': conversation_log['full_responses'][0:2],
#                 'final_answers': conversation_log['final_answers'][0:2],
#                 'prob_vectors': conversation_log['prob_vectors'][0:2],
#                 'calibrated_prob_vectors': [None], 
#                 'format_failures': conversation_log['format_failures'][0:2],
#                 'label': conversation_log['label'],
#             }
#             prior_conversation_logs[i].append(prior_conversation_log)
#         np.save(os.path.join(save_path, f"maze_{i}.npy"), prior_conversation_logs)


