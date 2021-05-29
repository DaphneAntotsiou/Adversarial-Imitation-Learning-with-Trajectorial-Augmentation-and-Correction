__author__ = 'DafniAntotsiou'

from baselines.common import set_global_seeds
import os
from cat_dauggi.tf_util import file_writer
import numpy as np


def csv_2_columnwise_dict(file):
    from collections import defaultdict
    from csv import DictReader

    columnwise_table = defaultdict(list)
    with open(file, 'rU') as f:
        reader = DictReader(f)
        for row in reader:
            for col, dat in row.items():
                columnwise_table[col].append(atof(dat))
    return columnwise_table


def atoi(text):
    return int(text) if text.replace('-', '', 1).isdigit() else text


def atof(text):
    return float(text) if text.replace('.', '', 1).replace('-', '', 1).isdigit() else text


def natural_keys(text):
    import re
    res = [atoi(c) for c in re.split('(\d+)', text) if c.isdigit()]
    return res


def evaluate_runs(func, func_list=None, func_dict=None, seed=0, keyword='hour', step=1,
                  stochastic=True, continue_hour=True, dir_separation=True):
    """
    :param func: function that runs the evaluation
    :param func_list: list parameters of func
    :param func_dict: dictionary parameters of func
    :param seed: seed of run
    :param keyword: keyword that will be excluded from the name search
    :param step: step between the runs
    :param stochastic: boolean for stochastic or deterministic policy run
    :param continue_hour: continue from the last evaluated step, if a previous run is available
    :param dir_separation: boolean if the checkpoints are separated in different subdirs w/ checkpoint files or not
    :return:
    """

    from decimal import Decimal

    if 'load_model_path' in func_dict and 'number_trajs' in func_dict:
        checkpoint_path = func_dict['load_model_path']
        trajectories = func_dict['number_trajs']
    else:
        print('No path to evaluate...')
        return

    network_prefix = func_dict['network_prefix'] if 'network_prefix' in func_dict else ''

    hour_path = os.path.join(checkpoint_path, keyword)
    hour_path = hour_path.replace("\\", "/")

    # add writer for log
    policy_type = "stochastic" if stochastic else "deterministic"
    logdir = os.path.join(checkpoint_path, keyword + "_eval_" + str(func_dict["number_trajs"]), "seed_" + str(seed),
                          policy_type)
    writer = file_writer(logdir)
    if writer:
        from cat_dauggi.statistics import stats
    success_stats = None

    paths = []
    if dir_separation:
        from glob import glob
        paths = glob(hour_path + "*")
        paths.sort(key=natural_keys)
    else:
        def get_step(elem):
            return elem.rstrip(".index").rpartition("_")[2]

        def step_int(elem):
            return int(get_step(elem))

        paths = sorted([os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))
                        and f.find(".index") != -1 and get_step(f).isnumeric()], key=step_int)

    # NOTE: BC is the same as hour000

    res = {}

    import csv
    res_file = logdir + '/res_' + network_prefix + '_' + str(trajectories) + '.csv'

    # find out if it should append or not
    start_hour = 0  # starting hour to compute
    if continue_hour and os.path.isfile(res_file):
        existing_res = csv_2_columnwise_dict(res_file)
        if existing_res["step"]:
            start_hour = max(existing_res["step"]) + 1

    mode = 'w' if not start_hour else 'a'

    import tensorflow as tf

    func_dict["find_checkpoint"] = True if dir_separation else False

    is_first = True
    for path in paths:
        path = path.replace("\\", "/")

        if path.find(hour_path) != -1 and dir_separation:
            hour = path.replace(hour_path, '')
            if hour.isdigit():
                hour = int(hour)
            else:
                # is not hour - probably eval
                continue
        elif not dir_separation:
            path = path.rstrip(".index")  # remove index
            hour = int(path.rpartition("_")[2])  # get the step number
        else:
            # no hour or BC in paths
            continue

        if hour < start_hour or (hour % step and hour != 155):  # add 155 for InvertedPendulum run
            continue

        print("evaluating hour ", hour)

        # reset seed so all runs have the same conditions
        set_global_seeds(seed)  # reset seed because of dataset randomisation
        if func_list is not None and len(func_list) > 0:
            func_list[0].seed(seed)
        elif "env" in func_dict:
            func_dict["env"].seed(seed)

        func_dict['load_model_path'] = path
        res[hour] = func(*func_list, **func_dict)

        # log iteration
        curr_checkpoint_file = tf.train.latest_checkpoint(path) if dir_separation else path
        if curr_checkpoint_file.rfind("iter_") != -1 and \
                curr_checkpoint_file[curr_checkpoint_file.rfind("iter_") + len("iter_"):].isdigit():
            curr_iter = int(curr_checkpoint_file[curr_checkpoint_file.rfind("iter_") + len("iter_"):])
        elif not dir_separation:
            curr_iter = hour
        else:
            curr_iter = None

        if is_first:
            # get name titles of statistics
            name_list = ['step', 'avg_len', 'avg_ret']

            if curr_iter is not None:
                name_list.append("iteration")

            if res and len(list(res.values())[0]) > 2 and isinstance(list(res.values())[0][2], dict):
                name_list.extend([v for v in list(res.values())[0][2].keys()])

            if writer and success_stats is None:
                success_stats = stats(name_list, scope="Evaluation")

            if mode == 'w':
                # write the titles
                with open(res_file, mode=mode, newline='') as hour_file:
                    csv_writer = csv.writer(hour_file)
                    csv_writer.writerow(name_list)
                mode = 'a'

        # get results of this hour

        result = res[hour]

        avg_len = result[0]
        avg_ret = result[1]
        prc_dict = result[2]

        res_list = [hour, avg_len, avg_ret]

        if curr_iter is not None:
            res_list.append(curr_iter)

        if prc_dict and isinstance(prc_dict, dict):
            res_list.extend([(str(v)) for v in prc_dict.values() if v is not None])

        if writer:
            success_stats.add_all_summary(writer, [float(v) for v in res_list], curr_iter if curr_iter is not None else hour)
            # # remove all the BC for the graph and make them -1
            # success_stats.add_all_summary(writer, [-1 if isinstance(v, str) else v for v in res_list],
            #                               -1 if isinstance(hour, str) else curr_iter if curr_iter is not None else hour)

        with open(res_file, mode=mode, newline='') as hour_file:
            # save as csv
            csv_writer = csv.writer(hour_file)
            csv_writer.writerow(res_list)

    # has evaluated all the hours
    print("evaluation of hourly results saved at: ", res_file)
    return res_file


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst)-1
    for i in range(start,-1,-1):
        if lst[i] == val:
            return i


def best_checkpoint_dir(log, dir, dir_separation=True, keyword=""):
    """
    Get best checkpoint based on the log from evaluate_hours
    :param log: the csv file with the log, produced by evaluate_hours
    :param dir: optional the directory with all the checkpoints. If None, is log parent
    :param keyword: optional keyword to filter the dir subforlders
    :return: checkpoint file with the best results in log
    """
    max_idx = None
    if os.path.isfile(log):
        log_dict = csv_2_columnwise_dict(log)
        key = 'success' if 'success' in log_dict else 'avg_ret'

        if key not in log_dict:
            print("No appropriate success key in csv. Terminating...")
            exit(1)

        max_idx = rindex(log_dict[key], max(log_dict[key]))
        if 'step' in log_dict:
            step = log_dict['step'][max_idx]
            max_idx = step
        else:
            print("no appropriate step key in csv. Terminating...")
            exit(1)
    else:
        print("log file does not exist. Terminating...")
        exit(1)

    if os.path.isdir(dir):

        if dir_separation:
            paths = [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o)) and o.find(keyword) != -1]
            paths.sort(key=natural_keys)

            for path in paths:
                if log.find(path) != -1:
                    continue    # is log path
                idx = natural_keys(os.path.basename(os.path.normpath(path)))
                if len(idx) and idx[0] == max_idx:
                    return path
        else:
            for f in os.listdir(dir):
                if os.path.isfile(f) and f.find(".index") != -1:
                    f = f.rstrip(".index")
                    if int(f.rpartition("_")[2]) == max_idx:
                        return f

        # should not reach this
        print("something went wrong... no best path found")
        exit(1)

    print("dir file does not exist. Terminating...")
    exit(1)


def evaluate_sigmas(step, low, high, func, func_list=None, func_dict=None, seed=0, stochastic=True):
    """
    Evaluate a checkpoint for different sigmas
    :param step:
    :param low:
    :param high:
    :param func:
    :param func_list:
    :param func_dict:
    :param seed:
    :param stochastic:
    :return:
    """

    if 'load_model_path' in func_dict and 'number_trajs' in func_dict:
        checkpoint_path = func_dict['load_model_path']
        trajectories = func_dict['number_trajs']
    else:
        print('No path to evaluate...')
        return


    network_prefix = func_dict['network_prefix'] if 'network_prefix' in func_dict else ''

    # add writer for log
    policy_type = "stochastic" if stochastic else "deterministic"
    logdir = os.path.join(checkpoint_path, "sigma_eval_" + str(func_dict["number_trajs"]), "seed_" + str(seed),
                          policy_type)
    writer = file_writer(logdir)
    if writer:
        from cat_dauggi.statistics import stats
    success_stats = None

    res = {}

    import csv
    res_file = logdir + '/res_' + network_prefix + '_' + str(trajectories) + '.csv'

    import tensorflow as tf

    mode = 'w'
    is_first = True
    for sigma in np.arange(low, high + step, step): # closed bounds
        print("evaluating sigma ", sigma)

        # reset seed so all runs have the same conditions
        set_global_seeds(seed)  # reset seed because of dataset randomisation
        if func_list is not None and len(func_list) > 0:
            func_list[0].seed(seed)
            func_list[0].env.env.sigma = sigma
        elif "env" in func_dict:
            func_dict["env"].seed(seed)
            func_dict["env"].env.env.sigma = sigma

        res[sigma] = func(*func_list, **func_dict)

        if is_first:
            # get name titles of statistics
            name_list = ['sigma', 'avg_len', 'avg_ret']

            if res and len(list(res.values())[0]) > 2 and isinstance(list(res.values())[0][2], dict):
                name_list.extend([v for v in list(res.values())[0][2].keys()])

            if writer and success_stats is None:
                success_stats = stats(name_list, scope="Sigma_evaluation")

            if mode == 'w':
                # write the titles
                with open(res_file, mode=mode, newline='') as hour_file:
                    csv_writer = csv.writer(hour_file)
                    csv_writer.writerow(name_list)
                mode = 'a'

        # get results of this sigma

        result = res[sigma]

        avg_len = result[0]
        avg_ret = result[1]
        prc_dict = result[2]

        res_list = [sigma, avg_len, avg_ret]

        if prc_dict and isinstance(prc_dict, dict):
            res_list.extend([(str(v)) for v in prc_dict.values() if v is not None])

        if writer:
            success_stats.add_all_summary(writer, [float(v) for v in res_list], sigma * 1000)   # *1000 because iter int

        with open(res_file, mode=mode, newline='') as sigma_file:
            # save as csv
            csv_writer = csv.writer(sigma_file)
            csv_writer.writerow(res_list)

    # has evaluated all the sigmas
    print("evaluation of sigma results saved at: ", res_file)
    return res_file
