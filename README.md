# Multilayer-Perceptron Classfier from Scratch

This is an implementation of Multilayer-Perceptron Classfier, which is an simplified version of sklearn's MLPClassfier. It also includes an GAOptimizer for the optimization of the neural network. Therefore, you can use Genetic Algorithm to optimize the neural network's parameters by setting the solver to 'ga'.

## Requirements

To use this classifier, you need:

```
python >= 3.7
numpy >= 1.19.2
scikit-learn >= 0.23.2
scipy >= 1.5.4
```

## Usage
You can try this MLP classifier using the following command:

```console
python3 mlp.py
```
It will load the provided daibetes dataset, and do 5-fold cross validation. This classifer's interfaces are quite similar to that of sklearn's, so you can use it just like using sklearn. 
