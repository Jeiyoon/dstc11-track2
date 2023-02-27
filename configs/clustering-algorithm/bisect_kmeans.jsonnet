local bisect(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'bisect',
    clustering_algorithm_params: {
        // n_init: 10,
        n_clusters: 10,
    },
};

bisect