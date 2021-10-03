import numpy as np


def model(theta_0: float, theta_1: float, data: np.ndarray):
    return theta_0 + theta_1 * data


def compute_mse(theta_0: float, theta_1: float, data: np.ndarray):
    """
    Calcula o erro quadratico medio
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """

    x = data[:, 0]
    y = data[:, 1]

    y_hat = model(theta_0, theta_1, x)

    mse = np.mean((y_hat - y)**2)

    return mse


def step_gradient(theta_0: float, theta_1: float, data: np.ndarray, alpha: float):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de theta_0 e theta_1.
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de theta_0 e theta_1, respectivamente
    """

    x = data[:, 0]
    y = data[:, 1]

    loss_theta_0 = np.mean(2 * (model(theta_0, theta_1, x) - y))
    loss_theta_1 = np.mean(2 * x * (model(theta_0, theta_1, x) - y))

    return theta_0 - alpha * loss_theta_0, theta_1 - alpha * loss_theta_1


def fit(data: np.ndarray, theta_0: float, theta_1: float, alpha: float, num_iterations: int):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de theta_0 e theta_1.
    Ao final, retorna duas listas, uma com os theta_0 e outra com os theta_1
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os theta_0 e outra com os theta_1 obtidos ao longo da execução
    """

    theta_0s = []
    theta_1s = []
    losses = []

    for epoch in range(num_iterations):
        print("Epoch %d" % epoch)
        theta_0s.append(theta_0)
        theta_1s.append(theta_1)
        losses.append(compute_mse(theta_0, theta_1, data))

        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha)

        print("\tMSE is %.5f" % losses[-1].item())

    return theta_0s, theta_1s
