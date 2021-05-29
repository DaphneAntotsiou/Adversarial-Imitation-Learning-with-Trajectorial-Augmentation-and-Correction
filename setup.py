from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='cat_dauggi',
      version='0.0.1',
      packages=['cat_dauggi', 'gym_ext', 'adroit_hand_ext'],
      install_requires=['mujoco-py==1.50.1.68', 'gym==0.12.0', 'baselines', 'scipy', 'filterpy',
                        'mj-envs', 'numpy<=1.14.4', 'fastdtw', 'tensorflow-gpu==1.12.0']
      )
