local birch(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'birch',
    clustering_algorithm_params: {
        // n_init: 10,
        n_clusters: 10,
    },
};

birch