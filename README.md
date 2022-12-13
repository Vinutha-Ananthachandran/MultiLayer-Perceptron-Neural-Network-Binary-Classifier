# MultiLayer-Perceptron-Neural-Network-Binary-Classifier
Implementation of a multi-hidden-layer neural network learner for XOR, Gaussian and Circle dataset, that will do the following. 
For a given dataset,
1. Constructs and trains a neural network classifier using provided labeled training data using nothing other than numpy library,
2. Use the learned classifier to classify the unlabeled test data,
3. Output the predictions of your classifier on the test data into a file in the same directory for XOR, Gaussian and Circle dataset

Model description -
1. The model implements a vanilla feed-forward neural network, possibly with 3 hidden layers. The network has 2 input nodes and outputs a single value.
2. Beyond this, there are no constraints on the model’s structure; 
3. It also uses cross-entropy as the loss function (each dataset presents a binary
classification task).
4. For back propogation, it uses gradient descent algorithm
