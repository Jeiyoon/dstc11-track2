local dbscan(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'dbscan',
    clustering_algorithm_params: {
        // eps: 0.3,
        // min_samples: 5,
        // n_init: 10,
        min_samples: 5,
    },
};

dbscan


