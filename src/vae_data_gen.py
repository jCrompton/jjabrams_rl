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

def get_n_training_data(n, data_dir='/Users/jamescrompton/PycharmProjects/jjabrams_rl/data/training_data/'):
    saved_images = os.listdir(data_dir)
    np.random.shuffle(saved_images)
    return np.array([scipy.misc.fromimage(Image.open('{}{}'.format(data_dir, path))) for path in saved_images[:n]])

def training_data_generator(batch_size, data_dir='/Users/jamescrompton/PycharmProjects/jjabrams_rl/data/training_data/'):
    while True:
        saved_images = os.listdir(data_dir)
        np.random.shuffle(saved_images)
        for i in range(0, len(saved_images), batch_size):
            data = np.array([scipy.misc.fromimage(Image.open('{}{}'.format(data_dir, path))) for path in saved_images[i:i+batch_size]])
            yield (data, data)

def gen_vae_data(game, samples_per_stage=10000, threads=2, size=(320,320), **kwargs):
    training_data_dir = '{}/PycharmProjects/jjabrams_rl/data/training_data/'.format(expanduser("~")) if not kwargs.get('data_dir') else kwargs.get('data_dir')
    p = multiprocessing.Pool(threads)
    gen_stage_data_args = [(game, stage, samples_per_stage, training_data_dir, size) for stage in retro.list_states(game)]
    data_added = sum(p.map(gen_stage_data, gen_stage_data_args))
    return data_added

def gen_stage_data(args):
    game, stage, samples, data_dir, size = args
    intro_skip = 0
    env = retro.make(game=game, state=stage)
    env.reset()
    states = 0
    done = False
    while not done and (states - intro_skip) <= samples:
        s, r, done, _ = env.step(env.action_space.sample())
        # Do not want the VAE to learn how to draw the opening screen...
        if states > intro_skip:
            img_name = '{}{}.jpg'.format(data_dir, uuid.uuid4())
            img = get_img_from_array(s, size=size)
            img.save(img_name)
        states += 1
    env.close()
    return (states-intro_skip-1)

def get_img_from_array(state, size=(320,320)):
    img = Image.fromarray(state)
    scaled_img = img.resize(size, Image.ANTIALIAS)
    return scaled_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=str, help='Name of game to play')
    parser.add_argument('--samples_per_stage', type=int, default=10000, help='Limit of images per stage (default 10000)')
    parser.add_argument('--threads', type=int, default=4, help='Number of cores to run collection on (default 4)')
    parser.add_argument('--size', type=tuple, default=(320,320), help='Size of image data, default VAE runs on 320x320 (confirm if doubtful)(default (320, 320))')
    parser.add_argument('--training_data_dir', type=str, default='/home/jamescrompton/jjabrams_rl/data/training_data/', help='Location of training data directory (default /home/jamescrompton/jjabrams_rl/data/training_data/)')
    args = parser.parse_args()

    print('Generating images for {} with {} samples per stage on {} threads with an image size of {} saving to {}...'.format(args.game, args.samples_per_stage, args.threads,
                                                                                                                             args.size, args.training_data_dir))
    images_data = gen_vae_data(args.game, samples_per_stage=args.samples_per_stage, threads=args.threads, size=args.size, data_dir=args.training_data_dir)
    total_data = len(os.listdir(args.training_data_dir))
    print("Generated {} images, saved to {} totalling to {} images.".format(images_data, args.training_data_dir, total_data))
