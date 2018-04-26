import retro
import os
import argparse
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
    scaled_img = img.resize(size, Image.ANTIALIAS)
    return scaled_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=str, help='Name of game to play')
    parser.add_argument('--samples_per_stage', type=int, default=10000, help='Limit of images per stage')
    parser.add_argument('--threads', type=int, default=4, help='Number of cores to run collection on')
    parser.add_argument('--size', type=tuple, default=(320,320), help='Size of image data, default VAE runs on 320x320 (confirm if doubtful)')
    parser.add_argument('--training_data_dir', type=str, default='/home/jamescrompton/jjabrams_rl/data/training_data/', help='Location of training data directory')
    args = parser.parse_args()

    return gen_vae_data(args.game, samples_per_stage=args.samples_per_stage, threads=args.threads, size=args.size, data_dir=args.training_data_dir)
