import retro
import os
import argparse
import h5py
import uuid
import random

from PIL import Image
import numpy as np
from vae_data_gen import training_data_generator, get_n_training_data

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, SeparableConv2D, BatchNormalization, Activation, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, TerminateOnNaN

from block_builder import Blocks

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_mean.shape[1]), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE:
    # Hyperparameters
    input_dim = (320, 320, 3)
    dense_size = 2048
    # Basic Encoder
    conv_filters = [32,64,64,128,128,256]
    conv_kernel_sizes = [4,4,4,4,4,4]
    conv_strides = [2,2,2,2,2,2]
    conv_activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    
    # RESNET
    res_net_arch0 = [(1,2,3,[64,64,256]), (1,3,3,[128,128,512]), (1,5,3,[256,256,1028]), (1,2,3,[512,512,2048])]
    res_net_arch = [(1,2,2, [32,32,64])]

    # Decoder
    conv_t_filters = [64, 64, 32, 32, 3]
    conv_t_kernel_sizes = [12, 12, 10, 8, 6]
    conv_t_strides = [2,2,2,2,2]
    conv_t_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

    z_dim = 128

    def __init__(self, *args, **kwargs):
        self.epochs = kwargs.get('epochs') if kwargs.get('epochs') else 1
        self.batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else 32
        self.block_builder = Blocks()
        self.model_weight_path = '/home/jamescrompton/'
        self.training_callbacks = [EarlyStopping(monitor='vae_kl_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'), TerminateOnNaN()]
        self.model, self.encoder, self.decoder = self._build(resnet_arch=kwargs.get('resnet_arch'))

    def _build_transpose_filter_cap(self, input_tensor):
        input_dim = self._get_decoder_cap_dim()
        

    def _get_decoder_cap_dim(self):
        dim = 1
        for kernel_size, stride in zip(self.conv_t_kernel_sizes, conv_t_strides):
            dim = stride*(dim-1) + kernel_size
        return dim

    def _build_basic_encoder(self, input_tensor):
        forward_loop = zip(self.conv_filters, self.conv_kernel_sizes, self.conv_strides, self.conv_activations)
        forward_input = input_tensor
        for i, arguments in enumerate(forward_loop):
            filter_size, kernel_size, strides, activation = arguments
            forward_input = SeparableConv2D(filters=filter_size, kernel_size=kernel_size,
                                strides=strides, activation=activation, name='CONV2D{}'.format(i))(forward_input)
            forward_input = BatchNormalization(axis=3, scale=False, name='BatchNorm{}'.format(i))(forward_input)
            forward_input = Activation('relu', name='Activation{}'.format(i))(forward_input)
        return forward_input

    def _build_resnet_encoder(self, input_tensor, pooling=(7,7)):
        resnet_block = input_tensor
        i = 0
        for convolutions, identities, kernel_size, filters in self.res_net_arch:
            resnet_block = self.block_builder.residual_block(resnet_block, kernel_size, filters, convolutions, identities, i)
            i+=1
        resnet_block = AveragePooling2D(pooling, name='avg_pool')(resnet_block)
        return resnet_block

    @staticmethod
    def vae_r_loss(y_true, y_pred):
            return 1000 * K.mean(K.square(y_true - y_pred), axis = [1,2,3])

    def _build(self, resnet_arch=False):
        # Encoder
        vae_input = Input(shape=self.input_dim, name='vae_input')
        forward_input = self._build_resnet_encoder(vae_input) if resnet_arch else self._build_basic_encoder(vae_input)

        # Compressed state vector
        vae_z_in = Flatten(name='vae_z_in')(forward_input)

        vae_z_mean = Dense(self.z_dim, name='vae_z_mean')(vae_z_in)
        vae_z_log_var = Dense(self.z_dim, name='vae_z_log_var')(vae_z_in)

        vae_z = Lambda(sampling, name='vae_z')([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(self.z_dim,), name='vae_z_input')

        vae_dense = Dense(self.dense_size, name='vae_dense')
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,self.dense_size), name='vae_z_out')
        vae_z_out_model = vae_z_out(vae_dense_model)

        # Decoder
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        backward_loop = zip(self.conv_t_filters, self.conv_t_kernel_sizes, self.conv_t_strides, self.conv_t_activations)
        backward_input = vae_z_out_model
        decoder_input = vae_z_out_decoder
        for i, arguments in enumerate(backward_loop):
            filter_size, kernel_size, strides, activation = arguments
            vae_dn = Conv2DTranspose(filters=filter_size, kernel_size=kernel_size, strides=strides, activation=activation, name='CONV2D_T{}'.format(i))
            backward_input = vae_dn(backward_input)
            decoder_input = vae_dn(decoder_input)

        # Reshape cap to fit OG image size
        backward_input = Reshape(self.input_dim, name='BackwardInputReshape')(backward_input)
        decoder_input = Reshape(self.input_dim, name='DecoderInputReshape')(decoder_input)
        # Make models
        vae = Model(vae_input, backward_input)
        vae_decoder = Model(vae_z_input, decoder_input)
        vae_encoder = Model(vae_input, vae_z)

        def vae_r_loss(y_true, y_pred):
            return 1000 * K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae.compile(optimizer='adam', loss='binary_crossentropy', metrics=[vae_r_loss, vae_kl_loss])
        # vae.compile(optimizer='adam', loss='binary_crossentropy')

        return(vae, vae_encoder, vae_decoder)

    def load_model(self, model_name):
        with open('../models/model_jsons/{}.json'.format(model_name), 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights('../models/weights/{}.h5'.format(model_name))
            print('Loaded model {} to object.model'.format(model_name))

    def save_model(self):
        if 'models' not in os.listdir('../'):
            print('Creating models directory...')
            os.mkdir('../models')
        if 'weights' not in os.listdir('../models/'):
            print('Creating weights directory...')
            os.mkdir('../models/weights')
        if 'model_jsons' not in os.listdir('../models/'):
            print('Creating model_jsons directory')
            os.mkdir('../models/model_jsons')
        # Generate unique name for models
        model_name = uuid.uuid4()
        print('Saving model with name {}...'.format(model_name))
        # Save model as JSON (save architecture)
        model_json = self.model.to_json()
        with open('../models/model_jsons/{}.json'.format(model_name), 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('../models/weights/{}.h5'.format(model_name))
        print('Saved {}'.format(model_name))

    def train_on_n(self, N, data_dir='/Users/jamescrompton/PycharmProjects/jjabrams_rl/data/training_data/', verbosity=2, shuffle=True):
        data = get_n_training_data(N, data_dir=data_dir)
        self.model.fit(data, data, epochs=self.epochs, callbacks=self.training_callbacks, batch_size=self.batch_size, shuffle=shuffle, verbose=verbosity)

    def gen_train(self, data_dir='/Users/jamescrompton/PycharmProjects/jjabrams_rl/data/training_data/', use_multiprocessing=False, workers=1, **kwargs):
        data_gen = training_data_generator(self.batch_size, data_dir=data_dir)
        steps_per_epoch = kwargs.get('steps_per_epoch') if kwargs.get('steps_per_epoch') else len(os.listdir(data_dir))/float(self.batch_size)
        try:
            self.model.fit_generator(data_gen, steps_per_epoch=steps_per_epoch, epochs=self.epochs, shuffle=True,
                                 callbacks=self.training_callbacks, use_multiprocessing=use_multiprocessing, workers=workers)
            self.save_model()
        except KeyboardInterrupt:
            print('Cancelling training and saving current model weights to file...')
            self.save_model()

    def predict(self, image_path):
        img = np.array(Image.open(image_path)).reshape((1,) + self.input_dim)
        pred_array = self.model.predict(img)
        print(pred_array[0])
        pred_img = Image.fromarray(pred_array[0])
        pred_img_name = uuid.uuid4()
        print(pred_img.shape)
        save_path = '{}/{}'.format('/'.join(path.split('/')[:-1], pred_img_name))
        print('Saving predicted image to given path: {}'.format(save_path))
        img.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=str, help='Name of game to play (eg SonicTheHedgehog2-Genesis)')
    parser.add_argument('--training_data_dir', type=str, default='/home/jamescrompton/jjabrams_rl/data/training_data/', help='Location of training data directory (default /home/jamescrompton/jjabrams_rl/data/training_data/)')
    parser.add_argument('--use_multiprocessing', type=bool, default=False, help='Use multiprocessing or not, (default False)')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers set only if multiprocessing True (default 1)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on (default 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (defaults to 32)')
    parser.add_argument('--resnet', action='store_true', help='If raised the encoder network will use the resnet architecture')
    parser.add_argument('--train', action="store_true", help='If raised the VAE will be trained using the above parameters (default True)')

    parser.add_argument('--prediction_image_path', type=str, default='', help='Path to image to run prediction on (default empty string)')
    parser.add_argument('--model_name', type=str, default='', help='Name of model to load (default empty string)')
    parser.add_argument('--predict', action="store_true", help='If raised the VAE will run a prediction on the specified image (default False), to be saved in the same directory as the given image.')

    args = parser.parse_args()
    if args.train:
        multiprocessing_str = ' multithreading with {} workers'.format(parser.workers) if args.use_multiprocessing else ''
        print('Training VAE on {} (epochs:{}, batch_size:{}) using pre-captured data from {}{}, CTRL-C to stop manually at any time...'.format(args.game, args.epochs, args.batch_size, args.training_data_dir, multiprocessing_str))

        vae = VAE(epochs=args.epochs, batch_size=args.batch_size, resnet_arch=args.resnet)
        vae.gen_train(data_dir=args.training_data_dir, use_multiprocessing=args.use_multiprocessing, workers=args.workers)
    elif args.predict:
        assert args.prediction_image_path != '', 'Argument --prediction_image_path cannot be an empty string, please specify the path to the image to run the VAE on.'
        assert args.model_name != '', 'Argument --model_name cannot be an empty string, please specify the name of the pre-trained model.'
        assert os.path.exists(args.prediction_image_path), 'Path to the prediction image does not exist, check you are entering the correct location'
        print('Predicting {} with {} weights'.format(args.prediction_image_path, args.model_name))

        vae = VAE()
        vae.load_model(args.model_name)
        vae.predict(args.prediction_image_path)
    else:
        print('You must specify either --train or --predict.')



