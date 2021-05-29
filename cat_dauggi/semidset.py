__author__ = 'DafniAntotsiou'

'''
 This implementation extends @openai/baselines.gail.dataset.mujoco_dset
 to include the semi-supervised auxiliary labels
'''

from cat_dauggi.mujoco_dset import Mujoco_Dset
import numpy as np


class SemiDset(Mujoco_Dset):

    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True, seed=0, semi_label='acs'):
        traj_data = dict(np.load(expert_path, allow_pickle=True))
        if semi_label not in traj_data:
            print("Semi supervised label not included in semi supervised dataset. Terminating...")
            exit(1)

        self._semi_label = semi_label

        self.info = expert_path  # save path for debugging purposes

        # add the semi-supervision labels
        self.semilabels = traj_data.copy()

        # change obs labels to include semi_label
        if semi_label in traj_data.keys() and 'obs' in traj_data and 'qpos' in traj_data and 'qvel' in traj_data:
            flat_obs = []
            for traj, ob_traj in zip(traj_data[semi_label], traj_data['obs']):
                flat_traj = []
                for frame, ob in zip(traj, ob_traj):
                    flat_traj.append(np.concatenate((frame.flatten(), ob)))
                flat_obs.append(np.asarray(flat_traj))
            traj_data['obs'] = np.asarray(flat_obs)
        else:
            print("ERROR missing labels for semi-supervised dataset")
            exit(1)

        super()._initialise(traj_data=traj_data, train_fraction=train_fraction, traj_limitation=traj_limitation,
                            randomize=randomize, seed=seed)

        self.semi_size = len(self.semilabels[semi_label][0][0]) if self.semilabels else 0
        self._traj_counter = 0                                  # counter to iterate trajectories
        self._frame_counter = 0                                 # counter to iterate frames in a trajectory
        self._total_traj = len(self.semilabels[semi_label])     # total number of semi-supervised trajectories

    def get_next_frame_full_ob(self, ob):
        """
        :return:
        the next full input and a bool if it's the last frame of the trajectory or not
        """

        done = False
        # input = np.concatenate((self.semilabels[self._traj_counter][self._frame_counter][self._semi_label], ob))
        input = np.concatenate((self.semilabels[self._semi_label][self._traj_counter][self._frame_counter], ob))
        self._frame_counter += 1
        if self._frame_counter == len(self.semilabels[self._semi_label][self._traj_counter]):
            done = True
            self._frame_counter = 0

        return input, done

    def full_ob_2_acs(self, full_ob):
        """
        Extract the action part from the full_ob vector
        :param full_ob: the full observations for the retargeting network
        :return: the actions part of the full observations
        """
        if self._semi_label != 'acs':
            print("WARNING: the semidataset actions are not environment actions for L2.")   # TODO: remove warning
        return np.copy(full_ob[:self.semi_size])

    def init_traj_labels(self, traj_id=-1, random_init=False):
        """
        return dictionary with the initial fully supervised labels of a trajectory (random in first 1/3 if traj_num = -1)
        the environment observations
        and the qvel and qpos of this frame
        returns None if traj_num is > available trajectories
        if random_init is False then only the first frame is used as fully supervised.
        """

        if traj_id > self._total_traj - 1:
            # requested trajectory doesn't exist - return None
            return None
        retries = 10
        while True:  # if trajectory is random, it shouldn't have length <= 1
            self._traj_counter = traj_id if traj_id >= 0 else self._randgen.randint(0, self._total_traj)
            # don't get the last frame
            self._frame_counter = 0 if not random_init else \
                self._randgen.randint(0, (len(self.semilabels[self._semi_label][self._traj_counter]) - 1) // 3)
            # get a random frame from the first third of the trajectory, remove // 3 for completely random frame
            if traj_id > 0 or not retries or len(self.semilabels[self._semi_label][self._traj_counter]) > 1:
                break
            retries -= 1

        if not retries:
            print("Couldn't find semi trajectory longer than 1 frame. Semi dataset is incompatible...")
            exit(1)

        # exclude the semi label from the return dictionary
        res = {label: self.semilabels[label][self._traj_counter][self._frame_counter]
               for label in self.semilabels.keys() if label != self._semi_label and
               isinstance(self.semilabels[label][self._traj_counter], np.ndarray)}
        # rename label of obs to ob
        res['ob'] = res.pop('obs')
        # add full_ob with the semi-labels
        res['full_ob'] = self.get_next_frame_full_ob(res['ob'])[0]      # returns tuple with done flag - ignore it
        return res

    @property
    def traj_id(self):
        return self._traj_counter

    @property
    def frame_counter(self):
        return self._frame_counter

    @frame_counter.setter
    def frame_counter(self, c):
        self._frame_counter = c

    @property
    def semi_trajectories(self):
        return self._total_traj