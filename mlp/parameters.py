"""
File defining all the parameters.

If you want to modify any of these default values, don't forget to change them both in `create_parameters` and
in the individual parameters below
"""

import os

DATA_SAVE_PATH = os.path.join("..", "saves", "data")
RESULT_SAVE_PATH = os.path.join("..", "saves", "results")
MODEL_SAVE_PATH = os.path.join("..", "saves", "models")

# Create missing directories
for path in [DATA_SAVE_PATH, RESULT_SAVE_PATH, MODEL_SAVE_PATH]:
    if not os.path.isdir(path):
        os.mkdir(path)

def parameters(unit: str = 'UMAR', to_normalize: bool = True,
                to_standardize: bool = False,
                name: str = "Default_model", save_path: str = MODEL_SAVE_PATH,
                seed: int = 0, batch_size: int = 128, shuffle: bool = True, n_epochs: int = 20,
                learning_rate: float = 0.05, decrease_learning_rate:bool = True,
                heating: bool = True, cooling: bool = True,
                warm_start_length: int = 12, minimum_sequence_length: int = 5, maximum_sequence_length: int = 240,
                overlapping_distance: int = 4, validation_percentage: float = 0.2, test_percentage: float = 0.,
                mlp_hidden_size: int = 256, mlp_num_layers: int = 1, division_factor: float = 10., activation_function: int = 0,
                verbose: int = 2):
    """
    Parameters of the models

    Returns:
        name:                       Name of the model
        save_path:                  Where to save models
        seed:                       To fix the seed for reproducibility
        heating:                    Whether to use the model for the heating season
        cooling:                    Whether to use the model for the cooling season
        room_models:                Which rooms to model
        learn_initial_hidden_states:Whether to learn the initial hidden and cell states
        warm_start_length:          Length of data to warm start the model (autoregression terms required to
                                      initialize hidden and cell states
        minimum_sequence_length:    Minimum length of a prediction sequence (forward)
        maximum_sequence_length:    Maximum length of a sequence to predict
        overlapping_distance:       Distance between overlapping sequences to predict
        batch_size:                 Batch size for the training of models
        shuffle:                    Flag to shuffle the data in training or testing procedure
        n_epochs:                   Number of epochs to train the model
        learning_rate:              Learning rate of the models
        decrease_learning_rate:     Flag to adjust the learning rate while training models
        validation_percentage:      Percentage of the data to put in the validation set
        test_percentage:            Percentage of the data to put in the test set
        feed_input_through_nn:      Flag whether to preprocess the input before the LSTM
        input_nn_hidden_sizes:      Hidden sizes of the NNs processing the input
        lstm_hidden_size:           Hidden size of the LSTMs processing the input
        lstm_num_layers:            Number of layers for the LSTMs
        layer_norm:                 Flag whether to put a normalization layer after the LSTMs
        output_nn_hidden_sizes:     Hidden sizes of the NNs processing the output
        division_factor:            Factor to scale the output of the networks to ease learning
        verbose:                    Verbose of the models
    """

    assert not (to_normalize and to_standardize), "Cannot normalize and standradize the data at the same time! " \
                                                "Please put either 'to_normalize' or 'to_standardize' to False."

    if not isinstance(division_factor, list):
        division_factor = [division_factor]

    return dict(name=name,
                save_path=save_path,
                seed=seed,
                unit=unit,
                to_normalize=to_normalize,
                to_standardize=to_standardize,
                heating=heating,
                cooling=cooling,
                warm_start_length=warm_start_length,
                minimum_sequence_length=minimum_sequence_length,
                maximum_sequence_length=maximum_sequence_length,
                overlapping_distance=overlapping_distance,
                batch_size=batch_size,
                shuffle=shuffle,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                decrease_learning_rate=decrease_learning_rate,
                validation_percentage=validation_percentage,
                test_percentage=test_percentage,
                mlp_hidden_size=mlp_hidden_size,
                mlp_num_layers=mlp_num_layers,
                division_factor=division_factor,
                activation_function=activation_function,
                verbose=verbose)

