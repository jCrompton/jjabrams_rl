import retro
import os
import multiprocessing
import numpy as np
import scipy.misc
import uuid
from os.path import expanduser
import random
from PIL import Image

def training_data_generator(game, batch_size, data_dir='/Users/jamescrompton/PycharmProjects/jjabrams_rl/data/training_data/'):
    saved_images = os.listdir(data_dir)
    np.random.shuffle(saved_images)
    for i in range(0, len(saved_images), batch_size):
        data = np.array([scipy.misc.fromimage(Image.open('{}{}'.format(data_dir, path))) for path in saved_images[i:i+batch_size]])
        yield data, data   

def gen_vae_data(game, samples_per_stage=10000, threads=2, size=(320,320), **kwargs):
    training_data_dir = '{}/PycharmProjects/jjabrams_rl/data/training_data/'.format(expanduser("~")) if not kwargs.get('data_dir') else kwargs.get('data_dir')
    p = multiprocessing.Pool(threads)
    gen_stage_data_args = [(game, stage, samples_per_stage, training_data_dir, size) for stage in retro.list_states(game)]
    return p.map(gen_stage_data, gen_stage_data_args)

def gen_stage_data(args):
    game, stage, samples, data_dir, size = args
    env = retro.make(game=game, state=stage)
    states = 0
    done = False
    while not done or states <= samples:
        s, r, done, _ = env.step(env.action_space.sample())
        img = get_img_from_array(s, size=size)
        img.save('{}{}.jpg'.format(data_dir, uuid.uuid4()))
        states += 1
    return states

def get_img_from_array(state, size=(320,320)):
    img = Image.fromarray(state)
    scaled_img = img.resize(size, Image.ANTIALIAS))
    return scaled_img