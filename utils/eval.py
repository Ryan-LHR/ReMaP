import warnings
from collections import defaultdict, OrderedDict

from termcolor import colored

from utils.other_utils import color_print


def get_apfd(pri_correct):
    """get APFD metric"""
    total_inputs = len(pri_correct)
    total_faults = pri_correct.count(False)
    faults_positions = [(index + 1) for index, value in enumerate(pri_correct) if not value]

    denominator = total_inputs * total_faults
    # Check if the denominator is zero
    if denominator == 0:
        apfd = -1
        apfd = 0
    else:
        apfd = 1 - (sum(faults_positions) / denominator) + (1 / (2 * total_inputs))
    return apfd


def get_rauc(pri_correct, budget=None):
    """get RAUC metric (mathematically verified)"""
    if budget != None:
        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_size = int(len(pri_correct) * budget)
        elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
            budget_size = budget
        pri_correct = pri_correct[:budget_size]

    total_inputs = len(pri_correct)
    total_faults = pri_correct.count(False)
    n = 0  # n_i
    N = 0  # sum of n_i

    for index, value in enumerate(pri_correct):
        if value == False:
            n += 1
        N += n

    # Check if the denominator is zero
    if total_faults == 0:
        rauc = -1
        rauc = 0
    else:
        rauc = N / (total_inputs * total_faults + (total_faults ** 2 - total_faults)/2)
    return rauc

def get_trc(pri_correct, budget=None, global_faults=-999):
    """get TRC metric"""
    if budget != None:
        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_size = int(len(pri_correct) * budget)
        elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
            budget_size = budget
        total_faults = pri_correct.count(False)  # for
    else:
        budget_size = int(len(pri_correct))
        total_faults = global_faults  # for SELECTION_METHOD

    if budget_size == 0:
        raise ValueError("Budget is 0")

    selected_correct = pri_correct[:budget_size]
    selected_faults = selected_correct.count(False)

    # Check if the denominator is zero
    if total_faults == 0:
        trc = -1
        trc = 0
    else:
        trc = selected_faults / min(budget_size, total_faults)
    return trc

def get_fdr(pri_correct, budget=None):
    """get FDR metric"""
    if budget != None:
        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_size = int(len(pri_correct) * budget)
        elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
            budget_size = budget
    else:
        budget_size = int(len(pri_correct))

    selected_correct = pri_correct[:budget_size]
    selected_faults = selected_correct.count(False)

    fdr = selected_faults / budget_size

    return fdr

def eval_overall(pri_correct, budget_list):
    """
    Evaluate all metrics (given a Global pri_correct with all samples)

    Args:
        pri_correct (list(Tensor(bool))): correct of prioritized candidate set

    """
    apfd = get_apfd(pri_correct)
    rauc = get_rauc(pri_correct)
    result_dict = {'APFD': round(apfd, 5),
                   'RAUC': round(rauc, 5),
                   }
    global_faults = pri_correct.count(False)

    for budget in budget_list:
        # get budget size
        if budget != None:
            if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
                budget_size = int(len(pri_correct) * budget)
                budget_percent = f"{budget * 100}%"
            elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
                budget_size = budget
        curr_pri_correct = pri_correct[:budget_size]

        # eval by each metric
        apfd = get_apfd(curr_pri_correct)
        rauc = get_rauc(curr_pri_correct)
        trc = get_trc(curr_pri_correct, None, global_faults)
        fdr = get_fdr(curr_pri_correct)

        result_dict.update({
            f'APFD-{budget_percent}': round(apfd, 5),
            f'RAUC-{budget_percent}': round(rauc, 5),
            f'TRC-{budget_percent}': round(trc, 5),
            f'FDR-{budget_percent}': round(fdr, 5),
        })
    return result_dict

def eval_overall_selection(prioritized_dict, cand_correct):
    # eval_overall(prioritized_indices, cand_correct)
    """
    Evaluate all metrics (given a set of pri_correct with different budget)

    Args:
        prioritized_dict (dict(budget: pri_correct)): correct of prioritized selected candidate set
    """

    result_dict = {}
    global_faults = cand_correct.count(False)

    for budget, prioritized_indices in prioritized_dict.items():
        if len(prioritized_indices) != len(set(prioritized_indices)):
            warnings.warn(
                f"prioritized indices of budget {budget} contains duplicate elements!!!",
                UserWarning
            )
            continue

        pri_correct = [cand_correct[i] for i in prioritized_indices]

        apfd = get_apfd(pri_correct)
        rauc = get_rauc(pri_correct)
        trc = get_trc(pri_correct, None, global_faults)
        fdr = get_fdr(pri_correct)

        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_percent = f"{budget * 100}%"

        result_dict.update({
            f'APFD-{budget_percent}': round(apfd, 5),
            f'RAUC-{budget_percent}': round(rauc, 5),
            f'TRC-{budget_percent}': round(trc, 5),
            f'FDR-{budget_percent}': round(fdr, 5),
        })

    return result_dict

def get_fault_types(pred_labels, true_labels, budget):
    """get fault types"""
    if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
        budget_size = int(len(pred_labels) * budget)
    elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
        budget_size = budget

    selected_pred = pred_labels[:budget_size]
    selected_true = true_labels[:budget_size]

    fault_types_dict = defaultdict(int)

    for true, pred in zip(selected_true, selected_pred):
        if true != pred:
            fault_types_dict[(true, pred)] += 1

    fault_types_dict = dict(fault_types_dict)
    fault_types = len(fault_types_dict)

    return fault_types

def eval_fault_types(pred_labels, true_labels, budget_list):
    """
    Evaluate fault types (Global)
    """
    num_classes = len(set(true_labels))
    total_types = num_classes * (num_classes - 1)

    print(f"\nTotal classes: {num_classes}"
          f"\nTotal fault types: {total_types}")

    fault_types_dict = {}
    for budget in budget_list:
        if budget != None:
            if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
                budget_percent = f"{budget * 100}%"
            elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
                ratio = float(budget / len(true_labels))
                budget_percent = f"{ratio * 100}%"

        fault_types = get_fault_types(pred_labels, true_labels, budget)
        fault_types_dict.update({f'Fault_Types-{budget_percent}': fault_types})

    return fault_types_dict

def eval_fault_types_selection(prioritized_dict, cand_labels, cand_truths):
    """
    Evaluate fault types (Selection based)
    """
    num_classes = len(set(cand_truths))
    total_types = num_classes * (num_classes - 1)

    print(f"\nTotal classes: {num_classes}"
          f"\nTotal fault types: {total_types}")

    fault_types_dict = {}
    for budget, prioritized_indices in prioritized_dict.items():
        pri_pred_labels = [cand_labels[i].item() for i in prioritized_indices]
        pri_true_labels = [cand_truths[i] for i in prioritized_indices]
        fault_types = get_fault_types(pri_pred_labels, pri_true_labels, 1.0)

        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_percent = f"{budget * 100}%"

        fault_types_dict.update({
            f'Fault_Types-{budget_percent}': fault_types,
        })
    return fault_types_dict

def eval_retrain(budget, acc_orig, acc_retrained, selected_correct):
    """
    Evaluate retrain metrics
    """

    if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
        budget_percent = f"{budget * 100}%"
    elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
        raise ValueError("Not Considered")
        # ratio = float(budget / len(true_labels))
        # budget_percent = f"{ratio * 100}%"

    acc_improved = acc_retrained - acc_orig

    num_wrong = sum(
        not c.item() for c in selected_correct
    )
    if hasattr(num_wrong, "item"):
        num_wrong = int(num_wrong.item())

    result_dict = {
        'Acc_Orig': round(acc_orig, 5) * 100,
        f'Acc_New-{budget_percent}': round(acc_retrained, 5) * 100,
        f'Acc_Diff-{budget_percent}': round(acc_improved, 5) * 100,
        f'Num_Wrong-{budget_percent}': num_wrong,
    }

    return result_dict

def print_results(rq, method, result_dict, seed=None):
    """
    Print results by group
    (return a ordered result dict)
    """

    # print(f'\nThe result of [{method.upper()}] for experiment [{rq.upper()}] is:')
    # color_print(f'\nThe result of [{method.upper()}] for experiment [{rq.upper()}] is:',
    #             'red', attrs=['bold', 'underline'])

    print(colored(f'\nThe result of', attrs=['bold', 'underline']),
          colored(f'{method.upper()}', 'red', attrs=['bold', 'underline']),
          colored(f'for experiment', attrs=['bold', 'underline']),
          colored(f'{rq.upper()}', 'red', attrs=['bold', 'underline']),
          colored(f'is:', attrs=['bold', 'underline']), end='\n')

    # add seed in result
    if seed != None:
        seed_dict = {'Seed': str(seed)}
        seed_dict.update(result_dict)
        result_dict = seed_dict

    # group by prefix before '-'
    groups = defaultdict(dict)
    for key, value in result_dict.items():
        if '-' in key:
            prefix = key.split('-')[0]
        else:
            prefix = key
        groups[prefix][key] = value

    sorted_dict = OrderedDict()
    # print every group (order in the meantime)
    for group, metrics in groups.items():
        print(" " * 6 + "{" , end='')
        color_print(f'{group}: ', 'red', end='')
        for k, v in metrics.items():
            # if 'Fault_Types' in k:
            #     print(f"'{k}': {v}, ", end='')
            if not isinstance(v, float):
                print(f"'{k}': {v}, ", end='')
                sorted_dict[k] = v
            elif 'Acc' in k:
                print(f"'{k}': {v:.2f}%, ", end='')
                sorted_dict[k] = round(v, 3)
            else:
                print(f"'{k}': {v:.5f}, ", end='')
                sorted_dict[k] = round(v, 5)
        print("}")

    return sorted_dict

def measure_execution_time(start_time, end_time, stage='Execution Time'):
    """
    Measure the execution time of a code block
    """
    elapsed_time = end_time - start_time
    elapsed_time = round(elapsed_time, 5)

    # convert time to h/min/s
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    exec_time_str = f"{hours}h {minutes}m {seconds:.2f}s"

    return {stage: (exec_time_str, elapsed_time)}

def average_time(time_list):
    """calculate average time"""
    total_seconds = 0

    for time_str in time_list:
        # extract hour, min, sec
        hours, minutes, seconds = 0, 0, 0
        time_parts = time_str.split()
        for part in time_parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))
            elif 's' in part:
                seconds = float(part.replace('s', ''))

        # convert time to total seconds
        total_seconds += hours * 3600 + minutes * 60 + seconds

    # average seconds
    avg_seconds = total_seconds / len(time_list)
    total_avg_seconds = round(avg_seconds, 2)

    # convert time to (hours, min, sec)
    avg_hours = int(avg_seconds // 3600)
    avg_seconds %= 3600
    avg_minutes = int(avg_seconds // 60)
    avg_seconds %= 60
    avg_seconds = round(avg_seconds, 2)

    time_24h = f"{avg_hours}h {avg_minutes}m {avg_seconds}s"
    return time_24h

def get_result_overall(result_list):
    """Calculate average performance of multiple result_dict"""

    # Initialize the overall result dictionary
    result_dict_overall = {'OverAll Result': 'As follows:'}
    num_exp = len(result_list)

    # Calculate average for numerical keys
    for key, value in result_list[0].items():
        if isinstance(value, (int, float)):
            result_dict_overall[key] = round(
                sum(d.get(key, 0) for d in result_list) / num_exp, 5
            )
        elif key == 'Execution Time':
            time_str_list = []
            for d in result_list:
                time_tuple = d.get('Execution Time', None)
                str_part, float_part = time_tuple
                time_str_list.append(str_part)
            # get average time in (hour, min, sec)
            avg_time_str = average_time(time_str_list)
            # get average time in seconds
            avg_time_float = round(
                sum(d.get(key, 0)[1] for d in result_list) / num_exp, 5
            )
            result_dict_overall[key] = (avg_time_str, avg_time_float)

        else:
            result_dict_overall[key] = None  # For non-numerical keys, set to None

    return result_dict_overall