# https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
# Load the iris data
from sklearn import datasets
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
# X: ndarray(500, 64)
# Y: ndarray(500, )
X = digits.data[:500]
y = digits.target[:500]

# Fit and transform with a TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

# Project the data in 2D
# X_2d: ndarray(500, 2)
X_2d = tsne.fit_transform(X)

# Visualize the data
# digits.target_names: ndarray(10, ) <- [0 to 9]
# target_ids: range(0, 10)
target_ids = range(len(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()