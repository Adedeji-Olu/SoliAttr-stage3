| Method                         | Accuracy   | MacroF1    |   Runs |
|:-------------------------------|:-----------|:-----------|-------:|
| B3 Majority class              | 9.1 ± 0.0  | 1.5 ± 0.0  |      5 |
| B1a PCA + Gaussian Bayes       | 64.1 ± 1.8 | 63.5 ± 1.9 |      5 |
| B1b RBF-SVM (pooled spectral)  | 81.2 ± 1.0 | 80.9 ± 1.0 |      5 |
| B2 CNN-RNN (Wang et al. style) | 91.2 ± 1.2 | 91.2 ± 1.2 |      5 |
| Project CNN (dt-map)           | 89.4 ± 0.5 | 89.0 ± 0.6 |      5 |
| RBF-SVM, random split (leaky)  | 93.8 ± 1.2 | -          |      5 |