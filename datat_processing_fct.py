import os
import json
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd

def load_experiment_data(
    root_dir: str,
    training_method: Union[str, List[str]],
    json_type: str = 'regular',  # 'regular' or 'CMA', regular represents non-CMA data (sigmas, means) files
    sections: List[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load selected sections of data from JSON files based on training method, experiment, and file type.

    Returns a dictionary: {person_id: [data_dicts]}
    """
    data = {}

    if isinstance(training_method, str):
        training_method = [training_method]

    if len(training_method) > 1:
        raise Exception("load_experiment_data expects one training method at a time")

    for method in training_method:
        method_path = os.path.join(root_dir, 'data', method)
        if not os.path.isdir(method_path):
            continue

        for person_id in os.listdir(method_path):
            person_path = os.path.join(method_path, person_id)
            if not os.path.isdir(person_path):
                continue
        
            files = [f for f in os.listdir(person_path) if f.endswith(".json")]
        
            person_data = []

            if json_type == 'CMA':
                cma_files = sorted([f for f in files if 'CMA' in f])
            
                if cma_files:
                    latest_cma_file = cma_files[-1]
                    with open(os.path.join(person_path, latest_cma_file), 'r') as f:
                        json_content = json.load(f)
                        if sections:
                            filtered = {k: json_content[k] for k in sections if k in json_content}
                        else:
                            filtered = json_content
                        person_data.append(filtered)

            elif json_type == 'regular':
                regular_files = [f for f in files if 'CMA' not in f]
                for file in regular_files:
                    with open(os.path.join(person_path, file), 'r') as f:
                        json_content = json.load(f)
                        if sections:
                            # filtered = {k: json_content[k] for k in sections if k in json_content}
                            filtered = json_content
                        else:
                            filtered = json_content
                        person_data.append(filtered)

            if person_data:
                data[person_id] = person_data

    return data


def normalize_heading_targets(targets: list, sentinel: float = 999999.0) -> float:
    """
    Given a flat list `targets` which uses `sentinel` to mark the start of each heading block,
    normalize each blocks values by its heading angle, average within block, then
    average those block means to a single scalar.
    """
    block_means = []
    i = 0
    n = len(targets)
    while i < n:
        # look for the sentinel
        if targets[i] == sentinel:
            # next element is the heading angle
            heading = targets[i + 1]
            # collect and normalize all values until the next sentinel
            vals = []
            j = i + 1
            while j < n and targets[j] != sentinel:
                vals.append(targets[j] / heading)
                j += 1
            if vals:  # only if we found some values to average
                block_means.append(np.mean(np.abs(vals))) # AVG ERROR
            i = j  # jump forward to continue scanning
        else:
            i += 1
    # print(len(block_means), block_means)
    if not block_means:
        raise ValueError("No heading blocks found in targets.")
    # finally average the per‐block means
    return float(np.mean(np.abs(block_means)))

def apply_normalization(
    data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Apply normalization to the specified sections of the data.
    """
    for person_id, trials in data.items():
        for trial in trials:
            tgt = trial.get('target', [])
            try:
                avg_norm = normalize_heading_targets(tgt)
            except ValueError:
                avg_norm = np.nan
            # store the result back in your trial dict:
            trial['normalized_heading_avg'] = avg_norm
    return data

# Reference parameter sets
reference_params = [
    [8.38, 0.8, 1.62],
    [8.38, 0.4, 0.88],
    [3.12, 0.8, 1.62],
    [3.12, 0.4, 0.88]
]

def is_param_close(param):
    return any(np.allclose(param, ref, rtol=1e-2, atol=1e-2) for ref in reference_params)

def filter_evaluation(data):
    data2 = {}
    for person_id, trials in data.items():
        filtered_trials = [trial for trial in trials if is_param_close(trial['params'])]
        data2[person_id] = filtered_trials
    return data2

def filter_training(data):
    data2 = {}
    for person_id, trials in data.items():
        filtered_trials = [trial for trial in trials if not is_param_close(trial['params'])]
        data2[person_id] = filtered_trials
    return data2

def selected_indices(n = 188, one_based=False):
        """
        Pattern:
        start: take 4
        then repeat 4x:  skip 3, take 4, skip 15, take 4
        then repeat 2x:  skip 16, take 4, skip 16, take 4
        Stops automatically if it would run past n.
        Returns 0-based by default; set one_based=True for 1-based.
        """
        idxs = []
        i = 0  # pointer into 0..n-1

        def take(k):
            nonlocal i
            for _ in range(k):
                if i >= n: 
                    return False
                idxs.append(i if not one_based else i + 1)
                i += 1
            return True

        def skip(k):
            nonlocal i
            i += k
            return i < n

        # start: take 4
        if not take(4):
            return idxs

        # 4 times: (skip 3, take 4, skip 15, take 4)
        for _ in range(4):
            if not (skip(3) and take(4) and skip(15) and take(4)):
                return [x for x in idxs if (x-1 if one_based else x) < n]

        # 2 times: (skip 16, take 4, skip 16, take 4)
        for _ in range(2):
            if not (skip(16) and take(4) and skip(16) and take(4)):
                break

        return idxs

def filter_evaluation_var(data_var):
    n_var = 188
    selected = selected_indices(n_var)
    selected_arr = np.asarray(selected, dtype=int)  # zero-based indices
    data_var_eval = {name: np.asarray(arr)[selected_arr]
                for name, arr in data_var.items()}
    return data_var_eval

def filter_training_var(data_var):
    n_var = 188
    selected = selected_indices(n_var)
    nonselected = [i for i in range(n_var) if i not in selected]
    nonselected_arr = np.asarray(nonselected, dtype=int)
    data_var_train = {name: np.asarray(arr)[nonselected_arr]
                for name, arr in data_var.items()}
    return data_var_train

def filter_evaluation_constant(data_constant):
    n_const = 244
    selected = [i for i in range(n_const) if i % 20 < 4]
    selected_arr = np.asarray(selected, dtype=int)  # zero-based indices

    def safe_take(arr, idx):
        a = np.asarray(arr)
        idx = idx[(idx >= 0) & (idx < a.shape[0])]
        # Take eval from 236-240 if needed, since all blocks in const are eval blocks
        if len(idx) == 48:
            idx = np.append(idx, [236,237,238,239])
        return a[idx]

    data_constant_eval = {name: safe_take(arr, selected_arr)
                for name, arr in data_constant.items()}
    return data_constant_eval

def filter_training_constant(data_constant):
    n_const = 244
    selected = [i for i in range(n_const) if i % 20 < 4]
    nonselected = [i for i in range(n_const) if i not in selected]
    nonselected_arr = np.asarray(nonselected, dtype=int)

    def safe_take(arr, idx):
        a = np.asarray(arr)
        idx = idx[(idx >= 0) & (idx < a.shape[0])]
        return a[idx]

    data_constant_train = {name: safe_take(arr, nonselected_arr)
                for name, arr in data_constant.items()}
    return data_constant_train

from typing import Dict, List, Any

def average_normalized_heading_by_fours(
    data: Dict[str, List[dict]],
    metric: str = "normalized_heading_avg",
    group_size: int = 4,
) -> Dict[str, List[float]]:
    """
    For each participant in `data`, compute the average of `metric` over
    non-overlapping groups of `group_size` consecutive trials.

    Parameters
    ----------
    data : dict
        Mapping {participant_name: list_of_trial_dicts}. Each trial dict must
        contain `metric` as a numeric value.
    metric : str
        The key to average from each trial dict. Default: 'normalized_heading_avg'.
    group_size : int
        The number of consecutive trials per group. Default: 4.

    Returns
    -------
    dict
        {participant_name: [avg_over_trials_0_3, avg_over_trials_4_7, ..., ]}
        For 52 trials and group_size=4, you'll get 13 values per participant.
    """
    result: Dict[str, List[float]] = {}

    for participant, trials in data.items():
        # Ensure we have an ordered list of trials
        if isinstance(trials, dict):
            # If trials are in a dict keyed by index, sort by numeric key
            ordered_trials = [trials[k] for k in sorted(trials, key=lambda x: int(x))]
        else:
            ordered_trials = list(trials)

        n = len(ordered_trials)
        n_groups = n // group_size  # ignore any leftover not filling a full group

        avgs: List[float] = []
        for g in range(n_groups):
            start = g * group_size
            chunk = ordered_trials[start:start + group_size]

            # Collect metric values, ignoring missing/non-numeric entries
            vals = []
            for t in chunk:
                if isinstance(t, dict) and metric in t:
                    v = t[metric]
                    if isinstance(v, (int, float)):
                        vals.append(float(v))

            # If all 4 present, average; if not, average what’s available
            # (still yields 13 values when n is 52)
            if vals:
                avgs.append(sum(vals) / len(vals))
            else:
                avgs.append(float("nan"))

        result[participant] = avgs

    return result

