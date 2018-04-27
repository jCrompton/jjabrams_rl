import argparse
from vae import VAE

def run_main(game, data_dir, use_multiprocessing, workers, *args, **kwargs):
    vae = VAE(**kwargs)
    vae.gen_train(data_dir=data_dir, use_multiprocessing=use_multiprocessing, workers=workers, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=str, help='Name of game to play')
    parser.add_argument('--training_data_dir', type=str, default='/home/jamescrompton/jjabrams_rl/data/training_data/', help='Location of training data directory (default /home/jamescrompton/jjabrams_rl/data/training_data/)')
    parser.add_argument('--use_multiprocessing', type=bool, default=False, help='Use multiprocessing or not, (default False)')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers set only if multiprocessing True (default 1)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on (default 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (defaults to 32)')

    args = parser.parse_args()
    multiprocessing_str = ' multithreading with {} workers'.format(parser.workers) if args.use_multiprocessing else ''
    print('Training VAE on {} (epochs:{}, batch_size:{}) using pre-captured data from {}{}...'.format(args.game, args.epochs, args.batch_size, args.training_data_dir, multiprocessing_str))
    run_main(args.game, args.training_data_dir, args.use_multiprocessing, args.workers, epochs=args.epochs, batch_size=args.batch_size)
