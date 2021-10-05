from typing import List, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt


def evaluate(individual: List[int]) -> int:
    """
    Recebe um indivíduo (lista de inteiros) e retorna o número de ataques
    entre rainhas na configuração especificada pelo indivíduo.
    Por exemplo, no individuo [2,2,4,8,1,6,3,4], o número de ataques é 10.

    :param individual:list
    :return:int numero de ataques entre rainhas no individuo recebido
    """

    num_attacks = 0
    on_board_coordinates = [(queen - 1, i) for i, queen in enumerate(individual)]

    for queen in on_board_coordinates:
        for potential_aggressor in on_board_coordinates:
            has_row_attack = queen[0] == potential_aggressor[0]
            has_column_attack = queen[1] == potential_aggressor[1]
            has_diagonal_attack = np.abs(queen[0] - potential_aggressor[0]) == np.abs(queen[1] - potential_aggressor[1])

            if has_diagonal_attack or has_column_attack or has_row_attack:
                num_attacks += 1

    # Remove "self attacks" and remove duplicates
    return (num_attacks - len(individual)) // 2


def tournament(participants: List[List[int]]) -> List[int]:
    """
    Recebe uma lista com vários indivíduos e retorna o melhor deles, com relação
    ao numero de conflitos
    :param participants:list - lista de individuos
    :return:list melhor individuo da lista recebida
    """

    return participants[np.argmin([evaluate(individual) for individual in participants])]


def crossover(parent_1: List[int], parent_2: List[int], index: int) -> Tuple[List[int], List[int]]:
    """
    Realiza o crossover de um ponto: recebe dois indivíduos e o ponto de
    cruzamento (indice) a partir do qual os genes serão trocados. Retorna os
    dois indivíduos com o material genético trocado.
    Por exemplo, a chamada: crossover([2,4,7,4,8,5,5,2], [3,2,7,5,2,4,1,1], 3)
    deve retornar [2,4,7,5,2,4,1,1], [3,2,7,4,8,5,5,2].
    A ordem dos dois indivíduos retornados não é importante
    (o retorno [3,2,7,4,8,5,5,2], [2,4,7,5,2,4,1,1] também está correto).
    :param parent_1:list
    :param parent_2:list
    :param index:int
    :return:list,list
    """

    offspring_1 = parent_1[0:index] + parent_2[index:]
    offspring_2 = parent_2[0:index] + parent_1[index:]

    return offspring_1, offspring_2


def mutate(individual: List[int], m: float):
    """
    Recebe um indivíduo e a probabilidade de mutação (m).
    Caso random() < m, sorteia uma posição aleatória do indivíduo e
    coloca nela um número aleatório entre 1 e 8 (inclusive).
    :param individual:list
    :param m:int - probabilidade de mutacao
    :return:list - individuo apos mutacao (ou intacto, caso a prob. de mutacao nao seja satisfeita)
    """

    if random.random() < m:
        individual[random.randint(0, 7)] = random.randint(1, 8)

    return individual


def run_ga(g: int, n: int, k: int, m: float, e: bool, plot: bool = False) -> List[int]:
    """
    Executa o algoritmo genético e retorna o indivíduo com o menor número de ataques entre rainhas
    :param g:int - numero de gerações
    :param n:int - numero de individuos
    :param k:int - numero de participantes do torneio
    :param m:float - probabilidade de mutação (entre 0 e 1, inclusive)
    :param e:bool - se vai haver elitismo
    :param plot:bool - opcional, se vai plotar o progresso ou nao
    :return:list - melhor individuo encontrado
    """

    # Initialize with n random individuals
    population = [random.choices(range(1, 9), k=8) for _ in range(n)]

    plot_mean = []
    plot_min = []
    plot_max = []

    for generation in range(g):
        new_population = []

        if e:
            new_population.append(population[np.argmin([evaluate(individual) for individual in population])])

        while len(new_population) < n:
            parent_1 = tournament([population[i] for i in random.sample(range(0, n), k)])
            parent_2 = tournament([population[i] for i in random.sample(range(0, n), k)])

            offspring_1, offspring_2 = crossover(parent_1, parent_2, random.randint(0, 7))

            offspring_1 = mutate(offspring_1, m)
            offspring_2 = mutate(offspring_2, m)

            new_population.append(offspring_1)
            new_population.append(offspring_2)

        population = new_population

        if plot:
            evaluations = [evaluate(individual) for individual in population]

            plot_mean.append(np.mean(evaluations))
            plot_min.append(np.min(evaluations))
            plot_max.append(np.max(evaluations))

    if plot:
        plt.figure(figsize=(10, 6), dpi=300)

        plt.plot(list(range(0, g)), plot_mean, color="blue", linestyle="dashdot", label="Mean")
        plt.plot(list(range(0, g)), plot_min, color="green", linestyle="dashed", label="Min")
        plt.plot(list(range(0, g)), plot_max, color="red", linestyle="dotted", label="Max")

        plt.xlabel("Generation")
        plt.ylabel("Number of conflicts")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("ga.png")

    best_individual = population[np.argmin([evaluate(individual) for individual in population])]

    return best_individual
