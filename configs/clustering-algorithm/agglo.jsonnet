local agglo(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'agglo',
    clustering_algorithm_params: {
        // n_init: 10,
        n_clusters: 10,
    },
};

agglo