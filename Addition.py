from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    """Этот алгоритм взят из алгоритма DEAP eaSimple() algorithm, с той модификацией, что
    hall of fame используется для реализации механизма элитизма. Особи, содержащиеся в
    halloffame, непосредственно вводятся в следующее поколение и не подвергаются
    генетическим операторам отбора, скрещивания и мутации."""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame параметр не должен быть пустым!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # процесс смены поколений
    for gen in range(1, ngen + 1):

        # вместо того чтобы отбирать индивидуумов в количестве,
        # равном размеру популяции, мы отбираем их меньше на столько,
        # сколько индивидуумов находится в зале славы:
        offspring = toolbox.select(population, len(population) - hof_size)

        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # после применения генетических операторов индивидуумы добавляются
        # из зала славы в популяцию:
        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring

        # добавление статистики текущего поколения в logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook