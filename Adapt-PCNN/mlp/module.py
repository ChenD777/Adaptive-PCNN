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
    
class PC_MLP(nn.Module):
    def __init__(
            self,
            device,
            supply_T_column: int,
            supply_m_column: int,
            power_column: int,
            temperature_column: int,
            inputs_D: list,
            division_factor: list,
            activation_function: int,
            mlp_hidden_size: int,
            mlp_num_layers: int,
            parameter_scalings: dict,
    ):

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.supply_T_column = supply_T_column
        self.supply_m_column = supply_m_column
        self.power_column = power_column
        self.temperature_column = temperature_column
        self.inputs_D = inputs_D
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.activation_function = activation_function
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_layers = mlp_num_layers
        self.last_D = None
        
        self.supply_T_column = self.supply_T_column[0] if isinstance(self.supply_T_column, list) else self.supply_T_column
        self.supply_m_column = self.supply_m_column[0] if isinstance(self.supply_m_column, list) else self.supply_m_column
        self.power_column = self.power_column[0] if isinstance(self.power_column, list) else self.power_column
        self.temperature_column = self.temperature_column[0] if isinstance(self.temperature_column, list) else self.temperature_column
        
        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = torch.Tensor(parameter_scalings['a']).to(self.device)
        self.b_scaling = torch.Tensor(parameter_scalings['b']).to(self.device)
        self.c_scaling = torch.Tensor(parameter_scalings['c']).to(self.device)
        self.d_scaling = torch.Tensor(parameter_scalings['d']).to(self.device)

        # Build the models
        self._build_model()

    def _build_model(self) -> None:

        ## Initialization of the parameters of `E`
        self.a = PositiveLinear(1, 1, require_bias=False)
        self.b = PositiveLinear(1, 1, require_bias=False)
        self.c = PositiveLinear(1, 1, require_bias=False)
        self.d = PositiveLinear(1, 1, require_bias=False)

        layers = []
        input_size = len(self.inputs_D)

        # 定义根据activation_function选择的激活函数
        if self.activation_function == 0:
            activation = nn.ReLU()
        elif self.activation_function == 1:
            activation = nn.Sigmoid()
        elif self.activation_function == 2:
            activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function code: {}".format(self.activation_function))

        # 添加输入层到第一隐藏层的连接
        layers.append(nn.Linear(input_size, self.mlp_hidden_size))
        layers.append(activation)

        # 添加隐藏层
        for _ in range(1, self.mlp_num_layers):
            layers.append(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size))
            layers.append(activation)

        # 添加输出层
        layers.append(nn.Linear(self.mlp_hidden_size, 1))

        # 将列表中的所有层组合成一个Sequential模型
        self.model = nn.Sequential(*layers)
        
        
        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.Tensor, warm_start: bool = False):
        
        if warm_start:
            x = x_[:, self.inputs_D].clone()
            SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
            SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
            IT_P = x_[:, self.power_column].clone().unsqueeze(1)
            RA_T = x_[:, self.temperature_column].clone().unsqueeze(1)
            self.last_D = torch.zeros((x.shape[0], 1)).to(self.device)
        else:
            x = x_[:, self.inputs_D].clone()
            SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
            SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
            IT_P = x_[:, self.power_column].clone().unsqueeze(1)
            x[:, self.temperature_column] = self.last_D[:, 0]
            RA_T = self.last_D
        
        # E part
        Temp_Diff =  RA_T - SA_T
        cool_effect = self.a((Temp_Diff * SA_V)) / self.a_scaling
        heat_effect = self.b(IT_P) / self.b_scaling
        
        # D part
        temp = self.model(x)
        output = RA_T + temp / self.division_factor - cool_effect + heat_effect
        self.last_D = output.clone()
        return output
    
    @property
    def E_parameters(self):
        return [float(x._parameters['log_weight']) for x in [self.a, self.b, self.c, self.d]]
    

class Adapt_PC_MLP(nn.Module):
    def __init__(
            self,
            device,
            supply_T_column: int,
            supply_m_column: int,
            power_column: int,
            temperature_column: int,
            inputs_D: list,
            division_factor: list,
            activation_function: int,
            mlp_hidden_size: int,
            mlp_num_layers: int,
            parameter_scalings: dict,
    ):
        super().__init__()
        self.device = device
        self.supply_T_column = supply_T_column
        self.supply_m_column = supply_m_column
        self.power_column = power_column
        self.temperature_column = temperature_column
        self.inputs_D = inputs_D
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.activation_function = activation_function
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_layers = mlp_num_layers
        self.last_D = None
        
        self.supply_T_column = self.supply_T_column[0] if isinstance(self.supply_T_column, list) else self.supply_T_column
        self.supply_m_column = self.supply_m_column[0] if isinstance(self.supply_m_column, list) else self.supply_m_column
        self.power_column = self.power_column[0] if isinstance(self.power_column, list) else self.power_column
        self.temperature_column = self.temperature_column[0] if isinstance(self.temperature_column, list) else self.temperature_column
        
        self._build_model()

    def _build_model(self):
        layers = []
        input_size = len(self.inputs_D)

        if self.activation_function == 0:
            activation = nn.ReLU()
        elif self.activation_function == 1:
            activation = nn.Sigmoid()
        elif self.activation_function == 2:
            activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function code: {}".format(self.activation_function))

        layers.append(nn.Linear(input_size, self.mlp_hidden_size))
        layers.append(activation)

        for _ in range(1, self.mlp_num_layers):
            layers.append(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size))
            layers.append(activation)

        layers.append(nn.Linear(self.mlp_hidden_size, 1))
        self.model = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
        
        # Xavier initialization of all the weights of NNs are set to 1
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.Tensor, warm_start: bool = False):
        if warm_start:
            x = x_[:, self.inputs_D].clone()
            SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
            SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
            IT_P = x_[:, self.power_column].clone().unsqueeze(1)
            RA_T = x_[:, self.temperature_column].clone().unsqueeze(1)
            self.last_D = torch.zeros((x.shape[0], 1)).to(self.device)
        else:
            x = x_[:, self.inputs_D].clone()
            SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
            SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
            IT_P = x_[:, self.power_column].clone().unsqueeze(1)
            x[:, self.temperature_column] = self.last_D[:, 0]
            RA_T = self.last_D

        combined_input = torch.cat((SA_T, SA_V, IT_P, RA_T), dim=1)
        coeff_a_b = F.softplus(self.fc1(combined_input))
        coeff_a_b = F.softplus(self.fc2(coeff_a_b))
        coeff_a, coeff_b = coeff_a_b.split(1, dim=1)
        
        cool_effect = coeff_a * (RA_T - SA_T) *  SA_V / 100
        heat_effect = coeff_b * IT_P / 100
        
        temp = self.model(x)
        output = RA_T + temp / self.division_factor - cool_effect + heat_effect
        self.last_D = output.clone()
        return output