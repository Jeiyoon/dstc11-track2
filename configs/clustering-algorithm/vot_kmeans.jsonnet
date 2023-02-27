local vot_kmeans(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'vot_kmeans',
    clustering_algorithm_params: {
        n_init: 10,
    },
};

vot_kmeans