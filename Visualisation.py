from deap import base
from deap import creator
from deap import tools
import random
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import Chess
import Dop

# константы задачи
NUM_OF_QUEENS = 16 

# константы генетического алгоритма
POPULATION_SIZE = 100  # количество индивидуумов в популяции
MAX_GENERATIONS = 100  # максимальное количество поколений
HALL_OF_FAME_SIZE = 10  # Зал славы
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума

# зерно для генератора случайных чисел
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# создание экземпляра класса NQueens
nQueens = Chess.NQueens(NUM_OF_QUEENS)

# экземпляр класса base.Toolbox
toolbox = base.Toolbox()

# создание класса для описания значения приспособленности особей
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# создание класса для представления каждого индивидуума
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# определение функции для генерации случайных значений
toolbox.register("randomOrder", random.sample, range(len(nQueens)), len(nQueens))

# определение функции для генерации отдельного индивидуума
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# определение функции для создания начальной популяции
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# функция расчёта приспособленности
def getViolationsCount(individual):
    return nQueens.getViolationsCount(individual),

# вычисление приспособленности каждой особи на основе getViolationsCount
toolbox.register("evaluate", getViolationsCount)

# генетические операторы:

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0 / len(nQueens))
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(nQueens))


def main():
    # создание начальной популяции:
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # подготовка объектов статистики:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # определение объекта зала славы:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # выполнение га с элементами элитизма:
    population, logbook = Dop.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # вывод лучших результатов:

    print("- Индивидуумы в зале славы:")
    for i in range(HALL_OF_FAME_SIZE):
        print(hof.items[i], sep="\n")

    best = hof.items[0]
    print("-- Лучший индивидуум = ", best)
    print("-- Лучшая приспособленность = ", best.fitness.values[0])

    # график статистик:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Минимальная / Средняя приспособленность')
    plt.title('Зависимость минимальной и средней приспособленности от поколения')

    # вывод лучшего результата на шахматное поле:
    sns.set_style("whitegrid", {'axes.grid': False})
    nQueens.plotBoard(hof.items[0])
    plt.show()


if __name__ == "__main__":
    main()