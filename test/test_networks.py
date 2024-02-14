import os
import sys

import torch

import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'dist', 'pycache', 'test')
lore = __import__('imputlorer')

X = torch.rand([32, 10, 64])


def test_mlp():
    net = lore.neural_networks.MLP(in_features=64,
                                   out_features=5,
                                   hidden_layers=[32, 32])
    y = net(X)
    assert y.shape == (32, 10, 5)


def test_lstm():
    net = lore.neural_networks.LSTM(input_size=64,
                                    output_size=5,
                                    hidden_size=64,
                                    n_layers=1)
    y = net(X)
    assert y.shape == (32, 10, 5)
