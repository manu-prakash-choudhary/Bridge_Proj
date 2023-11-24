'''
This will be our python file responsible for training models of different
kinds. We will use this to train our models and save them to disk.
we will take arguments from the command line to determine what kind of model
we want to train. We will also take arguments to determine what kind of data
we want to train on.'''

# imports
import argparse
import os
import sys
# import time


import numpy as np
# import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist


# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='model type to train')
    parser.add_argument('--data', type=str, default='mnist', help='data to train on')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to train with')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='directory to save models to')
    parser.add_argument('--save_name', type=str, default=parser.parse_args().model+'_model', help='name to save model as')
    parser.add_argument('--save_freq', type=int, default=1, help='how often to save model')
    parser.add_argument('--save_weights_only', type=bool, default=False, help='whether to save only weights or entire model')
    parser.add_argument('--save_format', type=str, default='h5', help='format to save model in')
    parser.add_argument('--save_best_only', type=bool, default=False, help='whether to save only the best model')
    parser.add_argument('--save_monitor', type=str, default='val_loss', help='monitor to use for save_best_only')
    parser.add_argument('--save_mode', type=str, default='auto', help='mode to use for save_best_only')
    parser.add_argument('--save_verbose', type=int, default=1, help='verbosity mode for saving')
    parser.add_argument('--save_options', type=dict, default={}, help='options for saving')
    parser.add_argument('--save_load_weights_on_restart', type=bool, default=False, help='whether to load weights on restart')
    parser.add_argument('--save_initial_epoch', type=int, default=0, help='initial epoch to start training at')
    parser.add_argument('--save_steps_per_epoch', type=int, default=1, help='steps per epoch to start training at')
    parser.add_argument('--save_max_queue_size', type=int, default=10, help='max queue size for saving')
    parser.add_argument('--save_workers', type=int, default=1, help='number of workers for saving')
    parser.add_argument('--save_use_multiprocessing', type=bool, default=False, help='whether to use multiprocessing for saving')
    parser.add_argument('--save_compile', type=bool, default=True)

    if len(sys.argv) == 2:
        if sys.argv[1] == '--help':
            parser.print_help()
            sys.exit(1)

    return parser.parse_args()


# function to train model
def train_model(model, data, epochs, batch_size, save_dir, save_name,
                save_freq, save_weights_only, save_format, save_best_only,
                save_monitor, save_mode, save_verbose, save_options,
                save_load_weights_on_restart, save_initial_epoch,
                save_steps_per_epoch, save_max_queue_size, save_workers,
                save_use_multiprocessing, save_compile):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, save_name)

    callbacks = []
    # set save_options
    if save_options == {}:
        save_options = None
    if save_freq > 0:
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=save_path,
                         monitor=save_monitor, verbose=save_verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=save_mode, save_freq=save_freq,
                         options=save_options,
                         load_weights_on_restart=save_load_weights_on_restart,
                         initial_epoch=save_initial_epoch,
                         steps_per_epoch=save_steps_per_epoch,
                         max_queue_size=save_max_queue_size,
                         workers=save_workers,
                         use_multiprocessing=save_use_multiprocessing))
    if save_compile:
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=save_path,
                         monitor=save_monitor, verbose=save_verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=save_mode,
                         save_freq=save_freq,
                         options=save_options,
                         load_weights_on_restart=save_load_weights_on_restart,
                         initial_epoch=save_initial_epoch,
                         steps_per_epoch=save_steps_per_epoch,
                         max_queue_size=save_max_queue_size,
                         workers=save_workers,
                         use_multiprocessing=save_use_multiprocessing))

    # data contains (train_images, train_labels), (test_images, test_labels)
    # train model
    y = data['arr_1']
    x = data['arr_0']
    
    model.fit(x, y, epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, validation_split=0.2)


# function to load data from path given
def load_data(data_path):
    # load data
    if data_path == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    else:
        data = np.load('./test_data/train_data_ready.npz')
        # print(data.shape, ' <----data shape')

    # handle data cardinality
    if data_path == 'mnist':
        print("train_images shape:", train_images.shape)
        print("train_labels shape:", train_labels.shape)
        print("test_images shape:", test_images.shape)
        print("test_labels shape:", test_labels.shape)
        data = (train_images, train_labels), (test_images, test_labels)
    return data


# function to load model from path given
def load_model(model_name: str):
    # load model
    if model_name == 'cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == 'rnn':
        model = models.Sequential()
        model.add(layers.SimpleRNN(50, input_shape=(50, 1), return_sequences=True))
        model.add(layers.SimpleRNN(50))
        model.add(layers.Dense(1))

        model.compile(optimizer="adam", loss='mean_squared_error',
                      metrics=['mae'])

    elif model_name == 'lstm':
        model = models.Sequential()
        model.add(layers.LSTM(50, input_shape=[None, 1], return_sequences=True))
        model.add(layers.LSTM(50, input_shape=[None, 1]))
        model.add(layers.Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    elif model_name == 'gru':
        model = models.Sequential()
        model.add(layers.GRU(10, input_shape=[None, 1], return_sequences=True))
        model.add(layers.GRU(10, input_shape=[None, 1]))
        model.add(layers.Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')

    elif model_name == 'tcn':
        model = models.Sequential()
        model.add(layers.Input(shape=(None, 1)))
        model.add(layers.TCN(nb_filters=64, kernel_size=2, nb_stacks=1,
                             dilations=[1, 2, 4, 8, 16, 32], padding='causal',
                             use_skip_connections=True, dropout_rate=0.2,
                             return_sequences=False, activation='relu',
                             kernel_initializer='he_normal',
                             use_batch_norm=False, use_layer_norm=False,
                             use_weight_norm=True, name='tcn'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    elif model_name == 'transformer':
        model = models.Sequential()
        model.add(layers.Input(shape=(None, 1)))
        model.add(layers.Transformer(num_layers=2, d_model=512, num_heads=8,
                                     dff=2048, input_vocab_size=10000,
                                     maximum_position_encoding=10000,
                                     rate=0.1))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == 'DenseNet':
        model = models.Sequential()
        model.add(layers.DenseNet121(include_top=True, weights=None,
                                     input_tensor=None, input_shape=None,
                                     pooling=None, classes=1000))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    else:
        model = models.load_model(model_name)
    return model


def load_weights(model, weights_path):
    model.load_weights(weights_path)


def save_model(model, save_path, save_format):
    model.save(save_path, save_format=save_format)


def main():
    # parse arguments
    args = parse_args()

    # load data
    data = load_data(args.data)

    # handle data cardinality
    # handle_data_cardinality(data)

    # load model
    model = load_model(args.model)

    # train model
    train_model(model, data, args.epochs, args.batch_size, args.save_dir,
                args.save_name, args.save_freq, args.save_weights_only,
                args.save_format, args.save_best_only, args.save_monitor,
                args.save_mode, args.save_verbose, args.save_options,
                args.save_load_weights_on_restart, args.save_initial_epoch,
                args.save_steps_per_epoch, args.save_max_queue_size,
                args.save_workers, args.save_use_multiprocessing,
                args.save_compile)

    # evaluate model
    # eval_model(model)

    # visualize results of different models
    # visualize_results()

    # save model
    save_model(model, os.path.join(args.save_dir, args.save_name),
               args.save_format)


if __name__ == '__main__':
    main()
