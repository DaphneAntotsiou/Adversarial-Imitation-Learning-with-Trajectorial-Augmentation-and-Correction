__author__ = 'DafniAntotsiou'

import numpy as np
from copy import deepcopy


# native functions currently not working on windows
def get_joint_qpos(sim, name):
    addr = sim.model.get_joint_qpos_addr(name)
    if not isinstance(addr, tuple):
        return sim.data.qpos[addr]
    else:
        start_i, end_i = addr
        return sim.data.qpos[start_i:end_i]


def get_joint_qvel(sim, name):
    addr = sim.model.get_joint_qvel_addr(name)
    if not isinstance(addr, tuple):
        return sim.data.qvel[addr]
    else:
        start_i, end_i = addr
        return sim.data.qvel[start_i:end_i]


def set_joint_qpos(sim, name, value):
    addr = sim.model.get_joint_qpos_addr(name)
    if not isinstance(addr, tuple):
        sim.data.qpos[addr] = value
    else:
        start_i, end_i = addr
        value = np.array(value)
        assert value.shape == (end_i - start_i,), (
                "Value has incorrect shape %s: %s" % (name, value))
        sim.data.qpos[start_i:end_i] = value

    return sim


def get_joint_state(name, data):
    if name is not None and data is not None:
        try:
            # object exists
            obj_pos = deepcopy(data.get_joint_qpos(name))
            obj_vel = deepcopy(data.get_joint_qvel(name))
            return obj_pos, obj_vel
        except ValueError:
            pass
    return None


def read_npz(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, (list, dict, tuple)):
        res = data
    elif isinstance(data, np.lib.npyio.NpzFile):
        res = dict(data)
        data.close()
    else:
        print("incompatible type of data to unzip...")
        res = None
    return res
