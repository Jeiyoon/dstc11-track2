local vot(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'vot',
    clustering_algorithm_params: {
        n_init: 10,
    },
};

vot