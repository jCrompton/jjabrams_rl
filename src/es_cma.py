import numpy as np
from deap import algorithms, base, creator, tools, cma
from collections import namedtuple
from scipy import stats
from keras import backend as K
from keras.callbacks import TerminateOnNaN

def evolve_model(data, model_func, population_size=10, cv=3, metric=mean_squared_error, **kwargs):
    """
        Evolve model takes in data a model function and a population size and finds the best model on that data through a basic GA implemented in DEAP
        :param data: A dictionary containing the data {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}
        :param model_func: A function that when called yields a model instance (any sklearn model works, or can take custom functions)
        :param population_size: An integer defining the size of the starting population you want to evolve
        :param cv: Either an integer or a callable, integer being the amount of k-folds and a callable being a custom k-fold generator
        :param metric: A function that performs the metric calculation (just use sklearn.metrics)
        :param kwargs: All optional parameters maximize, verbosity, ngen, indpb, cxpb, mutpb, winners, epochs, callbacks, etc..
        :return:
        """
    
  x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
  algo = kwargs.get('algo')
  cross_val = list(KFold(n_splits=cv, shuffle=True).split(x_train)) if type(cv) == int else list(cv)
  transcriber = neural_network_transcriber
  maximize = False if not kwargs.get('maximize') else kwargs.get('maximize')
  epochs = 5 if not kwargs.get('epochs') else kwargs.get('epochs')
  final_activation = 'linear' if not kwargs.get('classification') else kwargs.get('classification')
  callbacks = [TerminateOnNaN()] if not kwargs.get('callbacks') else kwargs.get('callbacks')
  verbosity = kwargs.get('verbose')
  n_gens = 25 if not kwargs.get('generations_number') else kwargs.get('generations_number')
  mutation_rate = 0.5 if not kwargs.get('indpb') else kwargs.get('indpb')
  prob_crossover = 0.5 if not kwargs.get('gene_crossover_prob') else kwargs.get('gene_crossover_prob')
  prob_mutation = 0.25 if not kwargs.get('gene_mutation_prob') else kwargs.get('gene_mutation_prob')
  winners = population_size if not kwargs.get('winners') else kwargs.get('winners')
  tournament_size = 6 if not kwargs.get('tournament_size') else kwargs.get('tournament_size')
  
  stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
  stats_size = tools.Statistics(key=len)
  stats_cache = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
  halloffame = tools.HallOfFame(population_size)
  fit_to = (1.0,) if maximize else (-1.0,)
  Results = namedtuple('Results', 'cv_results_ best_estimator_ cache best_params_')
  
  creator.create('FitnessMax', base.Fitness, weights=fit_to)
  creator.create('Individual', list, fitness=creator.FitnessMax)
  
  stats_cache.register("avg", np.mean, axis=0)
  stats_cache.register("std", np.std, axis=0)
  stats_cache.register("min", np.min, axis=0)
  stats_cache.register("max", np.max, axis=0)
  
  toolbox = base.Toolbox()
  genome = transcribe_base([], gene_len=True)
  toolbox.register('transcriber', stats.bernoulli.rvs, 0.5)
  toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.transcriber, n=genome[-1])
  toolbox.register('mate', customCX, genome=genome)
  toolbox.register('mutate', tools.mutShuffleIndexes, indpb=mutation_rate)
  
  toolbox.register('select', tools.selTournament, tournsize=tournament_size)
  toolbox.register('population', tools.initRepeat, list, toolbox.individual)

  def eval_func(individual, ret=False):
    on_nan = -(10 ** 10) if maximize else (10 ** 10)
    scores = []
    try:
      predictors = y_train.shape[1]
    except IndexError:
      predictors = 1
      model_params = transcriber(individual)
      model_params.update(shape=x_train.shape[1], predictors=predictors, final_activation=final_activation)
      batch_size = model_params.pop('batch_size')
      for train_index, test_index in cross_val:
        xtrain_cvs, xtest_cvs = x_train[train_index], x_train[test_index]
        ytrain_cvs, ytest_cvs = y_train[train_index], y_train[test_index]
        model = model_func(**model_params)
        if ret:
          return model
        model.fit(xtrain_cvs, ytrain_cvs, epochs=epochs, callbacks=callbacks, batch_size=batch_size,verbose=0)
        pred = model.predict(xtest_cvs)
        try:
          score = metric(ytest_cvs, pred)
        except ValueError:
          score = on_nan
         scores.append(score)
                                                           
      return np.mean(scores),

  toolbox.register('evaluate', eval_func)

  evo_start = time.time()
  if algo == 'cma':
      if verbosity > 0:
          print('Running evolution with Covariance Matrix Adaptation Evolution Strategy')
      size = transcribe_base([], gene_len=True)[-1]
      population = toolbox.population(n=population_size)
      centroid = [random.randint(0, 1) for _ in range(size)] if not kwargs.get('cma_seed') else kwargs.get('cma_seed')
      strategy = cma.Strategy(centroid=centroid, sigma=5.0, lambda_=population_size)
      toolbox.register("generate", strategy.generate, creator.Individual)
      toolbox.register("update", strategy.update)
      pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=n_gens, stats=stats_cache, halloffame=halloffame,
                                                 verbose=True)
  else:
      if verbosity > 0:
          print('Running Evolution with Simple Evolution Strategy')
      population = toolbox.population(n=population_size)
      pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=prob_crossover, mutpb=prob_mutation, ngen=n_gens,
                                         stats=stats_cache, halloffame=halloffame, verbose=True)
  mean_fit_time = float((time.time() - evo_start) / (n_gens * population_size))

  # Evaluate some metrics for the run
  avg_fitness = [gen['avg'][0] for gen in logbook.chapters['fitness']]
  best_raw = tools.selBest(population, k=winners)
  best_params = map(transcriber, best_raw)
  best_individuals = [eval_func(indv, ret=True) for indv in best_raw]
  best_evolutionary_scores = map(eval_func, best_raw)
  record = stats_cache.compile(pop)
                                                                    
  # Store most relevant data in cv_results dict and the rest in a cache (may be useful for graphs)
  cv_results = {'params': str(best_params[0]), 'rank_test_score': [1], 'mean_test_score': np.mean(avg_fitness),
      'std_test_score': np.std(avg_fitness), 'mean_fit_time': mean_fit_time,
      'model_name': model_func.__name__}
  cache = {'pop': pop, 'logbook': logbook, 'stats': stats_cache, 'halloffame': halloffame, 'record': record,
      'best_evolutionary_scores': best_evolutionary_scores}

  return Results(cv_results_=cv_results, best_estimator_=best_individuals[0], cache=cache,
                 best_params_=best_params[0])


def random_indexer(d):
    return [random.randint(0, len(vals) - 1) for key, vals in d.items()]


def neural_network_transcriber(bitarray):
    keys = ['hidden_architecture', 'dropout', 'optimizer', 'activation', 'batch_norm', 'batch_size']
    return {key: val for key, val in zip(keys, transcribe_base(bitarray)[1:-1])}


# STOLEN FROM DEAP TO TWEAK
def cxTwoPoint(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.
        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
    size = min(len(ind1), len(ind2))
    try:
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
    except ValueError:
        return ind1, ind2
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2


def customCX(parent1, parent2, genome=None):
    child1, child2 = parent1, parent2
    hamming = [(genome[i], genome[i + 1]) for i in range(len(genome) - 1) if
               parent1[genome[i]:genome[i + 1]] != parent2[genome[i]:genome[i + 1]]]
               random.shuffle(hamming)
               for start, end in hamming[:len(hamming) / 2]:
                   child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]
               return child1, child2
