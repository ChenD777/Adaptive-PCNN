import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, require_bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.require_bias = require_bias
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.require_bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.log_weight, 0.)

    def forward(self, input):
        if self.require_bias:
            return nn.functional.linear(input, self.log_weight.exp()) + self.bias
        else:
            return nn.functional.linear(input, self.log_weight.exp())

    
class PC_LSTM(nn.Module):
    def __init__(
            self,
            device,
            rooms: list,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            supply_T_column: int,
            supply_m_column: int,
            power_column: int,
            temperature_column: int,
            division_factor: list,
            activation_function: int,
            parameter_scalings: dict,
    ):

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.rooms = rooms
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.supply_T_column = supply_T_column
        self.supply_m_column = supply_m_column
        self.power_column = power_column
        self.temperature_column = temperature_column
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.activation_function = activation_function
        
        self.supply_T_column = self.supply_T_column[0] if isinstance(self.supply_T_column, list) else self.supply_T_column
        self.supply_m_column = self.supply_m_column[0] if isinstance(self.supply_m_column, list) else self.supply_m_column
        self.power_column = self.power_column[0] if isinstance(self.power_column, list) else self.power_column
        self.temperature_column = self.temperature_column[0] if isinstance(self.temperature_column, list) else self.temperature_column
        
        # Define latent variables
        self.last = None

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = torch.Tensor(parameter_scalings['a']).to(self.device)
        self.b_scaling = torch.Tensor(parameter_scalings['b']).to(self.device)
        self.c_scaling = torch.Tensor(parameter_scalings['c']).to(self.device)
        self.d_scaling = torch.Tensor(parameter_scalings['d']).to(self.device)

        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        
        # 根据activation_function选择的激活函数
        if self.activation_function == 0:
            activation = nn.ReLU()
        elif self.activation_function == 1:
            activation = nn.Sigmoid()
        elif self.activation_function == 2:
            activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function code: {}".format(self.activation_function))
            
        ## Initialization of the parameters of `E`
        self.a = PositiveLinear(1, 1, require_bias=False)
        self.b = PositiveLinear(1, 1, require_bias=False)
        self.c = PositiveLinear(1, 1, require_bias=False)
        self.d = PositiveLinear(1, 1, require_bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.inputs_D)] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), activation)
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.inputs_D)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules 
        # ensure the last layer has size 1 since we only model one zone
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.rooms)]
        self.output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), activation)
                                                for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        # print("LSTM Structure:")
        # print(self.input_nn)
        # print(self.lstm)
        # print(self.output_nn)

    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = torch.stack([self.initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                c = torch.stack([self.initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last
        else:
            self.last = torch.zeros((x.shape[0], len(self.rooms))).to(self.device)
            
        ## Forward 'D'
        if self.feed_input_through_nn:
            D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.inputs_D]
                for layer in self.input_nn:
                    temp = layer(temp)
                D_embedding[:, time_step, :] = temp
        else:
            D_embedding = x[:, :, self.inputs_D]
        
        lstm_output, (h, c) = self.lstm(D_embedding, (h, c))

        if self.layer_norm:
            lstm_output = self.norm(lstm_output)

        temp = lstm_output[:, -1, :]
        for layer in self.output_nn:
            temp = layer(temp)
        D = temp / self.division_factor
        
        ## Heat losses computation in 'E'
        # Cool by HVAC
        Temp_Diff_1 =  x[:, -1, self.temperature_column].clone() - x[:, -1, self.supply_T_column].clone()
        cool_effect = - self.a((Temp_Diff_1 * x[:, -1, self.supply_m_column].clone()).unsqueeze(1)) / self.a_scaling
        # print(f'self.a: {self.a(torch.tensor([1.0]).to(self.device))}')
        
        # Heat by HVAC
        P_1 = x[:, -1, self.power_column].clone().unsqueeze(1)
        heat_effect = self.b(P_1) / self.b_scaling
        
        E = cool_effect + heat_effect
        output = x[:, -1, [self.temperature_column]] + D + E
        self.last = output.clone()
        
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        return output, (h, c)

    @property
    def E_parameters(self):
        return [float(x._parameters['log_weight']) for x in [self.a, self.b, self.c, self.d]]
    
    
    
class Adapt_PC_LSTM(nn.Module):
    def __init__(
            self,
            device,
            rooms: list,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            supply_T_column: int,
            supply_m_column: int,
            power_column: int,
            temperature_column: int,
            division_factor: list,
            activation_function: int,
            parameter_scalings: dict,
    ):

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.rooms = rooms
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.supply_T_column = supply_T_column
        self.supply_m_column = supply_m_column
        self.power_column = power_column
        self.temperature_column = temperature_column
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.activation_function = activation_function
        
        self.supply_T_column = self.supply_T_column[0] if isinstance(self.supply_T_column, list) else self.supply_T_column
        self.supply_m_column = self.supply_m_column[0] if isinstance(self.supply_m_column, list) else self.supply_m_column
        self.power_column = self.power_column[0] if isinstance(self.power_column, list) else self.power_column
        self.temperature_column = self.temperature_column[0] if isinstance(self.temperature_column, list) else self.temperature_column
        
        # Define latent variables
        self.last = None
        
        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        
        # 根据activation_function选择的激活函数
        if self.activation_function == 0:
            activation = nn.ReLU()
        elif self.activation_function == 1:
            activation = nn.Sigmoid()
        elif self.activation_function == 2:
            activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function code: {}".format(self.activation_function))
        

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.inputs_D)] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), activation)
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.inputs_D)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules 
        # ensure the last layer has size 1 since we only model one zone
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.rooms)]
        self.output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), activation)
                                                for i in range(0, len(sizes) - 1)])
        
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
        
        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        # print("LSTM Structure:")
        # print(self.input_nn)
        # print(self.lstm)
        # print(self.output_nn)

    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = torch.stack([self.initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                c = torch.stack([self.initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last
        else:
            self.last = torch.zeros((x.shape[0], len(self.rooms))).to(self.device)
            
        ## Forward 'D'
        if self.feed_input_through_nn:
            D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.inputs_D]
                for layer in self.input_nn:
                    temp = layer(temp)
                D_embedding[:, time_step, :] = temp
        else:
            D_embedding = x[:, :, self.inputs_D]
        
        lstm_output, (h, c) = self.lstm(D_embedding, (h, c))

        if self.layer_norm:
            lstm_output = self.norm(lstm_output)

        temp = lstm_output[:, -1, :]
        for layer in self.output_nn:
            temp = layer(temp)
        D = temp / self.division_factor
        
        SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
        SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
        IT_P = x_[:, self.power_column].clone().unsqueeze(1)
        RA_T = x[:, -1, [self.temperature_column]].clone()
        
        combined_input = torch.cat((SA_T, SA_V, IT_P, RA_T), dim=1)
        coeff = F.softplus(self.fc1(combined_input))
        coeff = F.softplus(self.fc2(coeff))
        coeff_a, coeff_b  = coeff.split(1, dim=1)
        
        cool_effect = coeff_a * (RA_T - SA_T) *  SA_V / 100
        heat_effect = coeff_b * IT_P / 100
        
        E = - cool_effect + heat_effect
        output = x[:, -1, [self.temperature_column]] + D + E
        self.last = output.clone()
        
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        return output, (h, c)