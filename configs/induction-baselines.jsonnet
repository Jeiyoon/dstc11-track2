local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local agglo = import 'clustering-algorithm/agglo.jsonnet';

local tuned_agglo = tuned(
    clustering_algorithm = agglo(),
    parameter_search_space = {
        // No parameter -> IndexError: list index out of range
        n_clusters: ['quniform', 5, 50, 1]
        // min_samples: ['quniform', 5, 50, 1]

    },
    // k-means results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early stopping
    patience = 25,
);

{
    agglo: {name: 'agglo', model: tuned_agglo},
}