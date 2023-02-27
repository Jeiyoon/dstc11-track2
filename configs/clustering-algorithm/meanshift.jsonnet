local meanshift(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'meanshift',
    clustering_algorithm_params: {
        // n_init: 10,
        n_features_in: 10,
    },
};

meanshift