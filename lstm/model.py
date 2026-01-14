"""
File containing the base class of models, with general functions
"""

import os
import pandas as pd
import math
import time
from typing import Union

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import torch.nn.functional as F

from lstm.module import PC_LSTM, Adapt_PC_LSTM
from lstm.data import prepare_data
from lstm.util import model_save_name_factory, format_elapsed_time, inverse_normalize, inverse_standardize

class Model:
    """
    Class of models using PyTorch
    """

    def __init__(self, data: pd.DataFrame, interval: int, model_kwargs: dict, inputs_D: list, 
                module, rooms, supply_T_column: list = None, supply_m_column: list = None, power_column: list = None, 
                temperature_column: list = None, Y_columns: list = None, X_columns: list = None, topology: dict = None, 
                load_last: bool = False, load: bool = True):
        """
        Initialize a model.

        Args:
            model_kwargs:   Parameters of the models, see 'parameters.py'
            Y_columns:      Name of the columns that are to be predicted
            X_columns:      Sensors (columns) of the input data
        """
        assert module in ['PC_LSTM', 'Adapt_PC_LSTM'],\
                    f"The provided model type {module} does not exist, please chose among `'PC_LSTM', 'Adapt_PC_LSTM'`."
        
        # Define the main attributes
        self.name = model_kwargs["name"]
        self.model_kwargs = model_kwargs
        self.rooms = rooms if isinstance(rooms, list) else [rooms]

        # Create the name associated to the model
        self.save_name = model_save_name_factory(rooms=self.rooms, module=module, model_kwargs=model_kwargs)

        if not os.path.isdir(self.save_name):
            os.mkdir(self.save_name)

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])
        
        self.supply_T_column = supply_T_column if isinstance(supply_T_column, list) else [supply_T_column]
        self.supply_m_column = supply_m_column if isinstance(supply_m_column, list) else [supply_m_column]
        self.power_column = power_column if isinstance(power_column, list) else [power_column]
        self.temperature_column = temperature_column if isinstance(temperature_column, list) else [temperature_column]
        self.inputs_D = inputs_D
        self.topology = topology
        self.module = module

        self.unit = model_kwargs['unit']
        self.batch_size = model_kwargs["batch_size"]
        self.shuffle = model_kwargs["shuffle"]
        self.n_epochs = model_kwargs["n_epochs"]
        self.verbose = model_kwargs["verbose"]
        self.learning_rate = model_kwargs["learning_rate"]
        self.decrease_learning_rate = model_kwargs["decrease_learning_rate"]
        self.warm_start_length = model_kwargs["warm_start_length"]
        self.minimum_sequence_length = model_kwargs["minimum_sequence_length"]
        self.maximum_sequence_length = model_kwargs["maximum_sequence_length"]
        self.overlapping_distance = model_kwargs["overlapping_distance"]
        self.validation_percentage = model_kwargs["validation_percentage"]
        self.test_percentage = model_kwargs["test_percentage"]
        self.feed_input_through_nn = model_kwargs["feed_input_through_nn"]
        self.input_nn_hidden_sizes = model_kwargs["input_nn_hidden_sizes"]
        self.lstm_hidden_size = model_kwargs["lstm_hidden_size"]
        self.lstm_num_layers = model_kwargs["lstm_num_layers"]
        self.layer_norm = model_kwargs["layer_norm"]
        self.output_nn_hidden_sizes = model_kwargs["output_nn_hidden_sizes"]
        self.learn_initial_hidden_states = model_kwargs["learn_initial_hidden_states"]
        self.division_factor = model_kwargs['division_factor']
        self.activation_function = model_kwargs['activation_function']
        self.model_kwargs = model_kwargs
   
        # Prepare the data
        if self.verbose > 0:
            print(f'X_columns: {X_columns}, \n Y_columns: {Y_columns}')
        self.dataset = prepare_data(data=data, interval=interval, model_kwargs=model_kwargs, 
                                    Y_columns=Y_columns, X_columns=X_columns, verbose=self.verbose)
        
        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_losses = []
        self.validation_losses = []
        self._validation_losses = []
        self.test_losses = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.times = []
        self.sequences = []
        self.train_sequences = None
        self.validation_sequences = None
        self.test_sequences = None

        # Sanity check
        if self.verbose > 0:
            if self.module == "PhyInfo_LSTM":
                print('\nSanity check of the columns:\n', [(w, [self.dataset.X_columns[i] for i in x]) 
                        for w, x in zip(['Room temp', 'supply Temp', 'supply Mass'],
                                        [self.temperature_column, self.supply_T_column, self.supply_m_column])])
        if self.verbose > 0:
            print(f'self.dataset.X_columns: {self.dataset.X_columns}')
            print("Inputs used in D:\n", np.array(self.dataset.X_columns)[inputs_D])

        # To use the GPU when available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if self.verbose > 0:
                print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
            self.save_path = model_kwargs["save_path"]

        # Compute the scaled zero power points and the division factors to use in ResNet-like
        # modules
        self.parameter_scalings = self.create_scalings()

        # Prepare the torch module
        if self.module == "PC_LSTM":
            self.model = PC_LSTM(
                device=self.device,
                rooms=self.rooms,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                supply_T_column=self.supply_T_column,
                supply_m_column=self.supply_m_column,
                power_column=self.power_column,
                temperature_column=self.temperature_column,
                division_factor=self.division_factor,
                activation_function=self.activation_function,
                parameter_scalings=self.parameter_scalings,
            )  
        elif self.module == "Adapt_PC_LSTM":
            self.model = Adapt_PC_LSTM(
                device=self.device,
                rooms=self.rooms,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                supply_T_column=self.supply_T_column,
                supply_m_column=self.supply_m_column,
                power_column=self.power_column,
                temperature_column=self.temperature_column,
                division_factor=self.division_factor,
                activation_function=self.activation_function,
                parameter_scalings=self.parameter_scalings,
            )
        
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])
        self.loss = F.mse_loss

        # Load the model if it exists
        if load:
            self.load(load_last=load_last)

        # if the model doesn't exist, the sequences were not loaded
        if self.train_sequences is None:
            self.sequences = self.create_sequences()
            self.train_test_validation_separation(validation_percentage=self.validation_percentage, test_percentage=self.test_percentage)
            if self.verbose > 0:
                print(f'sequences: {len(self.sequences)}, train_sequences: {len(self.train_sequences)}, validation_sequences: {len(self.validation_sequences)}')

        self.model = self.model.to(self.device)

    @property
    def X(self):
        return self.dataset.X

    @property
    def Y(self):
        return self.dataset.Y

    @property
    def columns(self):
        return self.dataset.data.columns

    @property
    def show_sequences(self):
        return self.sequences, self.train_sequences, self.validation_sequences
    
    @property
    def show_max_min(self):
        max_x_value = self.X.max(axis=None)
        min_x_value = self.X.min(axis=None)
        max_y_value = self.Y.max(axis=None)
        min_y_value = self.Y.min(axis=None)

    def _fix_seeds(self, seed: int = None):
        """
        Function fixing the seeds for reproducibility.

        Args:
            seed:   Seed to fix everything
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_sequences(self, X: pd.DataFrame = None, Y: pd.DataFrame = None, inplace: bool = False):
        """
        Function to create tuple designing the beginning and end of sequences of data we can predict.
        This is needed because PyTorch models don't work with NaN values, so we need to only path
        sequences of data that don't contain any.

        Args:
            X:          input data
            Y:          output data, i.e. labels
            inplace:    Flag whether to do it in place or not

        Returns:
            The created sequences if not inplace.
        """

        # Take the data of the current model if nothing is given
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        # Get the places of NaN values not supported by PyTorch models
        nans = list(set(np.where(np.isnan(X))[0]) | set(np.where(np.isnan(Y))[0]))

        # List of indices that present no nans
        indices = np.arange(len(X))
        not_nans_indices = np.delete(indices, nans)
        last = len(indices) - 1

        sequences = []

        if len(not_nans_indices) > 0:
            # Get the "jumps", i.e. where the the nan values appear
            jumps = np.concatenate([[True], np.diff(not_nans_indices) != 1, [True]])

            # Get the beginnings of all the sequences, correcting extreme values and adding 0 if needed
            beginnings = list(not_nans_indices[np.where(jumps[:-1])[0]])
            if 0 in beginnings:
                beginnings = beginnings[1:]
            if last in beginnings:
                beginnings = beginnings[:-1]
            if (0 in not_nans_indices) and (1 in not_nans_indices):
                beginnings = [0] + beginnings

            # Get the ends of all the sequences, correcting extreme values and adding the last value if needed
            ends = list(not_nans_indices[np.where(jumps[1:])[0]])
            if 0 in ends:
                ends = ends[1:]
            if last in ends:
                ends = ends[:-1]
            if (last in not_nans_indices) and (last - 1 in not_nans_indices):
                ends = ends + [last]

            # We should have the same number of series beginning and ending
            assert len(beginnings) == len(ends), "Something went wrong"

            # Bulk of the work: create starts and ends of sequences tuples
            for beginning, end in zip(beginnings, ends):
                # Add sequences from the start to the end, jumping with the wanted overlapping distance and ensuring
                # the required warm start length and minimum sequence length are respected
                self.sequences += [(beginning + self.overlapping_distance * x,
                               min(beginning + self.warm_start_length + self.maximum_sequence_length
                                   + self.overlapping_distance * x, end))
                    for x in range(math.ceil((end - beginning - self.warm_start_length
                                              - self.minimum_sequence_length) / self.overlapping_distance))]


        return self.sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2, test_percentage: float = 0.0) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the sequences of
        data points that are in the training/testing part.

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing

        Returns:
            Nothing, in place definition of all the indices
        """

        # Sanity checks: the given inputs are given as percentage between 0 and 1
        if 1 <= validation_percentage <= 100:
            validation_percentage /= 100
            if self.verbose > 0:
                print("The train-test-validation separation rescaled the validation_percentage between 0 and 1")
        if 1 <= test_percentage <= 100:
            test_percentage /= 100
            if self.verbose > 0:
                print("The train-test-validation separation rescaled the test_percentage between 0 and 1")

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        if self.verbose > 0:
            print("Creating training, validation and testing data")
        
        for sequences in [self.sequences]:
            if len(sequences) > 0:
                # Given the total number of sequences, define aproximate separations between training
                # validation and testing sets
                train_validation_sep = int((1 - test_percentage - validation_percentage) * len(sequences))
                validation_test_sep = int((1 - test_percentage) * len(sequences))

                # Prepare the lists

                self.train_sequences += sequences[:train_validation_sep]
                self.validation_sequences += sequences[train_validation_sep:validation_test_sep]
                self.test_sequences += sequences[validation_test_sep:]

    def create_scalings(self):

        parameter_scalings = {}

        parameter_scalings['a'] = [100]
        parameter_scalings['b'] = [100]
        parameter_scalings['c'] = [100]
        parameter_scalings['d'] = [100]
        
        return parameter_scalings
    
    
    def batch_iterator(self, iterator_type: str = "train", batch_size: int = None, shuffle: bool = True) -> None:
        """
        Function to create batches of the data with the wanted size, either for training,
        validation, or testing

        Args:
            iterator_type:  To know if this should handle training, validation or testing data
            batch_size:     Size of the batches
            shuffle:        Flag to shuffle the data before the batches creation

        Returns:
            Nothing, yields the batches to be then used in an iterator
        """

        # Firstly control that the training sequences exist - create them otherwise
        if self.train_sequences is None:
            self.train_test_validation_separation()
            if self.verbose > 0:
                print("The Data was not separated in train, validation and test --> the default 70%-20%-10% was used")

        # If no batch size is given, define it as the default one
        if batch_size is None:
            batch_size = self.batch_size

        # Copy the indices of the correct type (without first letter in case of caps)
        if "rain" in iterator_type:
            sequences = self.train_sequences
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences
        elif "est" in iterator_type:
            sequences = self.test_sequences
        else:
            raise ValueError(f"Unknown type of batch creation {iterator_type}")

        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(sequences) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield sequences[batch * batch_size: (batch + 1) * batch_size]

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given sequences of indices corresponding to a batch.

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            batch_x:    Batch input of the model
            batch_y:    Targets of the model, the temperature and the power
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences) == tuple:
            sequences = [sequences]

        # Iterate over the sequences to build the input in the right form
        input_tensor_list = [torch.Tensor(self.X[sequence[0]: sequence[1], :].copy()) for sequence in sequences]

        # Prepare the output for the temperature and power consumption
        output_tensor_list = [torch.Tensor(self.Y[sequence[0]: sequence[1], :].copy()) for sequence in
                                    sequences]


        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
            batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0)
        else:
            batch_x = input_tensor_list[0].view(1, input_tensor_list[0].shape[0], -1)
            batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], -1)

        # Return everything
        return batch_x.to(self.device), batch_y.to(self.device)

    def predict(self, sequences: Union[list, int] = None, data: torch.Tensor = None, mpc_mode: bool = False):
        """
        Function to predict batches of "sequences", i.e. it creates batches of input and output of the
        given sequences you want to predict and forwards them through the network

        Args:
            sequences:  Sequences of the data to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predictions and the true output
        """

        return_y = True
        if sequences is not None:
            # Ensure the given sequences are a list of list, not only one list
            if type(sequences) == tuple:
                sequences = [sequences]

            # Build the input and output
            batch_x, batch_y = self.build_input_output_from_sequences(sequences=sequences)

        elif data is not None:
            if isinstance(data, tuple):
                if len(data[0].shape) == 3:
                    batch_x = data[0].reshape(data[0].shape[0], data[0].shape[1], -1)
                    batch_y = data[1].reshape(data[0].shape[0], data[0].shape[1], len(self.rooms))
                else:
                    batch_x = data[0].reshape(1, data[0].shape[0], -1)
                    batch_y = data[1].reshape(1, data[0].shape[0], len(self.rooms))
            else:
                if len(data.shape) == 3:
                    batch_x = data.reshape(data.shape[0], data.shape[1], -1)
                else:
                    batch_x = data.reshape(1, data.shape[0], -1)
                return_y = False

        else:
            raise ValueError("Either sequences or data must be provided to the `predict` function")
        
        predictions = torch.zeros((batch_x.shape[0], batch_x.shape[1], len(self.rooms))).to(self.device)
        states = None

        # Iterate through the sequences of data to predict each step, replacing the true power and temperature
        # values with the predicted ones each time
        for i in range(batch_x.shape[1]):
            # Predict the next output and store it
            pred, states = self.model(batch_x[:, i, :], states, warm_start=i<self.warm_start_length)
            predictions[:, i, :] = pred
        
        if return_y:
            return predictions, batch_y
        else:
            return predictions

    def scale_back_predictions(self, sequences: Union[list, int] = None, data: torch.Tensor = None):
        """
        Function preparing the data for analyses: it predicts the wanted sequences and returns the scaled
        predictions and true_data

        Args:
            sequences:  Sequences to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)

        Returns:
            The predictions and the true data
        """

        # Compute the predictions and get the true data out of the GPU
        predictions, true_data = self.predict(sequences=sequences, data=data)
        predictions = predictions.cpu().detach().numpy()
        true_data = true_data.cpu().detach().numpy()

        # Reshape the data for consistency with the next part of the code if only one sequence is given
        if sequences is not None:
            # Reshape when only 1 sequence given
            if type(sequences) == tuple:
                sequences = [sequences]

        elif data is not None:
            sequences = [0]

        else:
            raise ValueError("Either sequences or data must be provided to the `scale_back_predictions` function")

        if len(predictions.shape) == 2:
            predictions = predictions.reshape(1, predictions.shape[0], -1)
            true_data = true_data.reshape(1, true_data.shape[0], -1)

        # Scale the data back
        cols = self.dataset.Y_columns[:-1]
        truth = true_data.reshape(true_data.shape[0], true_data.shape[1], -1)
        true = np.zeros_like(predictions)

        if self.dataset.is_normalized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :] = inverse_normalize(data=predictions[i, :, :],
                                                         min_=self.dataset.min_[self.dataset.Y_columns],
                                                         max_=self.dataset.max_[self.dataset.Y_columns])
                true[i, :, :] = inverse_normalize(data=truth[i, :, :],
                                                       min_=self.dataset.min_[self.dataset.Y_columns],
                                                       max_=self.dataset.max_[self.dataset.Y_columns])
        elif self.dataset.is_standardized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :] = inverse_standardize(data=predictions[i, :, :],
                                                           mean=self.dataset.mean[self.dataset.Y_columns],
                                                           std=self.dataset.std[self.dataset.Y_columns])
                true[i, :, :] = inverse_standardize(data=truth[i, :, :],
                                                         mean=self.dataset.mean[self.dataset.Y_columns],
                                                         std=self.dataset.std[self.dataset.Y_columns])

        return predictions, true

    def fit(self, n_epochs: int = None, print_each: int = 5) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            n_batches_print:  Control how many batches to print per epoch

        Returns:
            Nothing
        """

        self.times.append(time.time())

        if self.verbose > 0:
            print("\nTraining starts!")

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # Define the best loss, taking the best existing one or a very high loss
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf

        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)

        for epoch in range(trained_epochs, trained_epochs + n_epochs):

            if self.verbose > 0:
                print(f"\nTraining epoch {epoch + 1}...")

            # Start the training, define a list to retain the training losses along the way
            self.model.train()
            train_losses = []
            train_sizes = []

            # Adjust the learning rate if wanted
            if self.decrease_learning_rate:
                self.adjust_learning_rate(epoch=epoch)

            # Create training batches and run through them, using the batch_iterator function, which has to be defined
            # independently for each subclass, as different types of data are handled differently
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="train")):

                # Compute the loss of the batch and store it
                loss = self.compute_loss(batch_sequences)

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                #for p in self.model.named_parameters():
                #    if (p[1].grad is not None):
                #        print(p[0], ":", p[1].grad.norm())
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_losses.append(float(loss))
                train_sizes.append(len(batch_sequences))

                # Regularly print the current state of things
                if (self.verbose > 1) & (num_batch % print_each == print_each - 1):
                    print(f"Loss batch {num_batch + 1}: {float(loss):.2E}")

            # Compute the average loss of the training epoch and print it
            train_loss = sum([l*s for l,s in zip(train_losses, train_sizes)]) / sum(train_sizes)
            if self.verbose > 0:
                print(f"Average training loss after {epoch + 1} epochs: {train_loss:.2E}")
            self.train_losses.append(train_loss)

            # Start the validation, again defining a list to recall the losses
            if self.verbose > 0:
                print(f"Validation epoch {epoch + 1}...")
            validation_losses = []
            _validation_losses = []
            validation_sizes = []
            _validation_sizes = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="validation", batch_size=self.batch_size, shuffle=False)):

                # Compute the loss, in the torch.no_grad setting: we don't need the model to
                # compute and use gradients here, we are not training
                if 'PiNN' not in self.name:
                    self.model.eval()
                    with torch.no_grad():
                        loss = self.compute_loss(batch_sequences)
                        validation_losses.append(float(loss))
                        validation_sizes.append(len(batch_sequences))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % (print_each) == (print_each) - 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss):.2E}")

                else:
                    self.model.train()
                    loss = self.compute_loss(batch_sequences)
                    validation_losses.append(float(loss))
                    validation_sizes.append(len(batch_sequences))
                    # Regularly print the current state of things
                    if (self.verbose > 1) & (num_batch % (print_each//2) == (print_each//2) - 1):
                        print(f"Loss batch {num_batch + 1}: {float(loss):.2E}")
                    self.model.eval()
                    with torch.no_grad():
                        loss = self.compute_loss(batch_sequences)
                        _validation_losses.append(float(loss))
                        _validation_sizes.append(len(batch_sequences))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % (print_each//2) == (print_each//2) - 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss):.2E}")

            # Compute the average validation loss of the epoch and print it
            validation_loss = sum([l*s for l,s in zip(validation_losses, train_sizes)]) / sum(validation_sizes)
            self.validation_losses.append(validation_loss)
            if self.verbose > 0:
                print(f"Average validation loss after {epoch + 1} epochs: {validation_loss:.2E}")

            if 'PiNN' in self.name:
                _validation_loss = sum([l*s for l,s in zip(_validation_losses, train_sizes)]) / sum(_validation_sizes)
                self._validation_losses.append(_validation_loss)
                if self.verbose > 0:
                    print(f"Average accuracy validation loss after {epoch + 1} epochs: {_validation_loss:.2E}")

            # Timing information
            self.times.append(time.time())
            if self.verbose > 0:
                print(f"Time elapsed for the epoch: {format_elapsed_time(self.times[-2], self.times[-1])}"
                      f" - for a total training time of {format_elapsed_time(self.times[0], self.times[-1])}")

            # Save parameters
            if 'PhyInfo_LSTM' in self.module:
                p = self.model.E_parameters
                self.a.append(p[0])
                self.b.append(p[1])
                self.c.append(p[2])
                self.d.append(p[3])
            
            # Save last and possibly best model
            self.save(name_to_add="last", verbose=0)

            if validation_loss < best_loss:
                verbose = 1 if self.verbose > 0 else 0
                self.save(name_to_add="best", verbose=verbose)
                best_loss = validation_loss

        if self.verbose > 0:
            best_epoch = np.argmin([x for x in self.validation_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch + 1} after training for " f"{trained_epochs + n_epochs} epochs")

    def adjust_learning_rate(self, epoch: int) -> None:
        """
        Custom function to decrease the learning rate along the training

        Args:
            epoch:  Epoch of the training

        Returns:
            Nothing, modifies the optimizer in place
        """

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.98

    def compute_loss(self, sequences: list):
        """
        Custom function to compute the loss of a batch of sequences.

        Args:
            sequences: The sequences in the batch

        Returns:
            The loss
        """

        predictions, batch_y = self.predict(sequences=sequences)
        loss = self.loss(predictions, batch_y)
        return loss

    def save(self, name_to_add: str = None, verbose: int = 0):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis.

        Args:
            name_to_add:    Something to save a unique model

        Returns
            Nothing, everything is done in place and stored in the parameters
        """

        if verbose > 0:
            print(f"\nSaving the new {name_to_add} model!")

        if name_to_add is not None:
            save_name = os.path.join(self.save_name, f"{name_to_add}_model.pt")
        else:
            save_name = os.path.join(self.save_name, "model.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_sequences": self.train_sequences,
                "validation_sequences": self.validation_sequences,
                "test_sequences": self.test_sequences,
                "train_losses": self.train_losses,
                "validation_losses": self.validation_losses,
                "_validation_losses": self._validation_losses,
                "test_losses": self.test_losses,
                "times": self.times,
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "warm_start_length": self.warm_start_length,
                "maximum_sequence_length": self.maximum_sequence_length,
                "feed_input_through_nn": self.feed_input_through_nn,
                "input_nn_hidden_sizes": self.input_nn_hidden_sizes,
                "lstm_hidden_size": self.lstm_hidden_size,
                "lstm_num_layers": self.lstm_num_layers,
                "output_nn_hidden_sizes": self.output_nn_hidden_sizes,
                "activation_function": self.activation_function
            },
            save_name,
        )

    def load(self, load_last: bool = False):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        if load_last:
            save_name = os.path.join(self.save_name, "last_model.pt")
        else:
            save_name = os.path.join(self.save_name, "best_model.pt")

        if self.verbose > 0:
            print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(save_name), f"The file {save_name} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_sequences = checkpoint["train_sequences"]
            self.validation_sequences = checkpoint["validation_sequences"]
            self.test_sequences = checkpoint["test_sequences"]
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            self.test_losses = checkpoint["test_losses"]
            self.times = checkpoint["times"]
            self.a = checkpoint["a"]
            self.b = checkpoint["b"]
            self.c = checkpoint["c"]
            self.d = checkpoint["d"]
            self.warm_start_length = checkpoint["warm_start_length"]
            self.maximum_sequence_length = checkpoint["maximum_sequence_length"]
            self.feed_input_through_nn = checkpoint["feed_input_through_nn"]
            self.input_nn_hidden_sizes = checkpoint["input_nn_hidden_sizes"]
            self.lstm_hidden_size = checkpoint["lstm_hidden_size"]
            self.lstm_num_layers = checkpoint["lstm_num_layers"]
            self.output_nn_hidden_sizes = checkpoint["output_nn_hidden_sizes"]
            self.activation_function=checkpoint["activation_function"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")
                print(f"It contains {len(self.train_sequences)} training sequences and "
                      f"{len(self.validation_sequences)} validation sequences.\n")        
            
#             # 获取模型的 state_dict
#             model_state_dict = self.model.state_dict()

#             # 检查是否包含 MLP 模型的参数
#             mlp_keys = [key for key in model_state_dict.keys() if key.startswith('standalone_mlp')]
#             print("Keys related to MLP in the model's state_dict:", mlp_keys)
            
#             # 打印 MLP 模型的参数值
#             for key in mlp_keys:
#                 print(f"{key}: {model_state_dict[key]}")

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            if self.verbose > 0:
                print("\nNo existing model was found!\n")
