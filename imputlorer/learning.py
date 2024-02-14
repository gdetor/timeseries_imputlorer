import pickle

import torch
from torch import nn
from torch.optim import Adam

from ray import tune
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from .neural_networks import MLP


DEVICE = torch.device("cuda:0")


def objective(config):
    in_features = 1
    out_features = 1

    epochs = config["epochs"]
    lrate = config["lrate"]
    hidden_size = config["hidden_size"]
    n_layers = config["n_layers"]

    hidden_layers = [hidden_size for i in range(n_layers)]

    net = MLP(in_features=in_features,
              out_features=out_features,
              hidden_layers=hidden_layers)

    optimizer = Adam(net.parameters(), lr=lrate)
    criterion = nn.MSELoss()

    for e in range(epochs):
        net.train()
        for x, y in training_data:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            yhat = net(x)

            loss = criterion(yhat, y)
            loss.backward()

            optimizer.step()

        net.eval()
        with torch.no_grad():
            for x, y in testing_data:
                yhat = net(x)

    session.report({"accuracy": accuracy})


def optimize(data):
    search_space = {"epochs": tune.randint(50, 500),
                    "lrate": tune.loguniform(1e-5, 1e-2),
                    "n_layers": tune.randint(1, 3),
                    "hidden_size": tune.randint(16, 512)}

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=2)
    scheduler = AsyncHyperBandScheduler()

    tune_ = tune.with_parameters(objective, data=data)
    tune_with_resources_ = tune.with_resources(tune_,
                                               resources={"cpu": 5,
                                                          "gpu": 1})
    tune_conf = tune.TuneConfig(metric="accuracy",
                                mode="max",
                                search_alg=algo,
                                scheduler=scheduler,
                                num_samples=5)
    tuner = tune.Tuner(tune_with_resources_,
                       tune_config=tune_conf,
                       param_space=search_space)
    results = tuner.fit()
    print(results.get_best_result().config)
    with open("../results/best_params.pkl") as f:
        pickle.dump(results.get_best_result().config, f)
