__author__ = 'DafniAntotsiou'

'''
Auxiliary functions to help with creating/saving/loading tf models
'''

from baselines.common.tf_util import *
from tensorflow.python import pywrap_tensorflow

def file_writer(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return tf.summary.FileWriter(dir_path, get_session().graph)


def load_state(fname, var_list=None):
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(get_session(), fname)
    return True


def save_state(fname, var_list=None, write_meta_graph=True):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver(var_list=var_list)
    saver.save(get_session(), fname, write_meta_graph=write_meta_graph)


def save_no_prefix(fname, var_list, prefix='', write_meta_graph=True):
    ''' remove the prefix from the node names before saving'''

    if prefix:
        var_dict = {}
        for var in var_list:
            var_dict[var.op.name.replace(prefix, "")] = var
        save_state(fname, var_list=var_dict, write_meta_graph=write_meta_graph)  # save the variables without the prefix
    else:
        save_state(fname, var_list=var_list, write_meta_graph=write_meta_graph)  # save the variable list - no prefix present


def load_with_prefix(fname, var_list, prefix=''):
    ''' load nodes from checkpoint with prefix and remove it'''

    if prefix:
        # rename the variables from a standard architecture to this
        var_dict = {}
        for var in var_list:
            var_dict[prefix + var.op.name] = var
        load_state(fname, var_list=var_dict)
    else:
        load_state(fname, var_list=var_list)


def load_checkpoint_variables(fname, prefix='', check_prefix='', include_no_prefix_vars=True):
    '''
    load the variables from a checkpoint that match the current variables.
    Add a prefix to the names of the checkpoint if prefix is defined.
    Match the (network) prefix to a checkpoint prefix, if check_prefix is defined.
    First check if there are same names and load them if include_no_prefix_vars is True.
    '''
    if fname is not None and tf.train.checkpoint_exists(fname):
        reader = pywrap_tensorflow.NewCheckpointReader(fname)
        var_to_shape_map = reader.get_variable_to_shape_map()

        if include_no_prefix_vars:
            var_dict = {v.op.name: v for v in tf.global_variables()
                        if v.op.name in var_to_shape_map.keys()}
        else:
            var_dict = {}

        if prefix:
            # replace check_prefix with prefix for the active network
            var_dict.update({v.op.name.replace(prefix, check_prefix): v for v in tf.global_variables()
                             if v.op.name.replace(prefix, check_prefix) in var_to_shape_map.keys()
                             and v.op.name.find(prefix) != -1})
        elif check_prefix:
            # network has no prefix - add checkpoint's prefix to keys
            var_dict.update({check_prefix + v.op.name: v for v in tf.global_variables()
                             if check_prefix + v.op.name in var_to_shape_map.keys()})

        if not var_dict or all(name.find("adversary") != -1 for name in var_dict.keys()):
            print("checkpoint " + fname + " has no compatible variables with prefix " + check_prefix)
            return False
        # var_dict = {key: var_dict[key] for key in var_dict if key.find("adversary") == -1}

        invalid_keys = []
        for key in var_dict:
            if var_to_shape_map[key] != var_dict[key].shape.as_list():
                print("ERROR, key {} is {} in graph but {} in checkpoint. Returning...".format(
                    key, var_dict[key].shape.as_list(), var_to_shape_map[key]))
                invalid_keys.append(key)
        if invalid_keys:
            return False
        if not load_state(fname, var_list=var_dict):
            print("error loading "+ fname)
            return False
        ALREADY_INITIALIZED.update(var_dict.values())
        print("checkpoint " + fname + " loaded successfully")
        return True
    elif fname:
        msg = "WARNING: " + fname + " checkpoint does not exist..."
        print(msg)
    return False


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            # device_count={'GPU': 0}
        )
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)
