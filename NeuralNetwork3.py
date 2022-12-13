import sys
import numpy as np

random_seed = 2764763
np.random.seed(random_seed)

class StartLayer():
    def __init__(self, i_node_count):
        self.i_node_count = i_node_count
        self.shape = (self.i_node_count,)

    def forward_feed(self, istream):
        self.idata = istream
        self.odata = self.idata
        return self.odata

class HiddenLayer():
    def __init__(self, i_node_count, o_node_count, actv, actv_gradient):
        self.i_node_count = i_node_count
        self.o_node_count = o_node_count
        self.shape = (self.i_node_count, self.o_node_count)
        self.wghts = np.random.rand(i_node_count, o_node_count) - 0.5
        self.gradient = np.zeros_like(self.wghts)
        self.actv = actv
        self.actv_gradient = actv_gradient
        self.out_flag = False

    def forward_feed(self, istream):
        self.idata = istream
        self.z = np.matmul(np.transpose(self.wghts), self.idata)
        self.odata = self.actv(self.z)
        if self.out_flag == False:
            self.odata = np.concatenate(
                [np.array([1.0], ndmin=2), self.odata], axis=0)
        return self.odata

    def backward_propagation(self, istream, weights_data):
        self.difference = istream
        if self.out_flag == True:
            self.difference = self.odata - self.difference
        else:
            self.difference = np.matmul(weights_data, self.difference)
            self.difference = np.delete(self.difference, 0, axis=0)
            self.difference = self.difference * self.actv_gradient(self.z)

class MultiLayerPerceptron():
    def __init__(self, error, lrate):
        self.layers = []
        self.error = error
        self.lrate = lrate

    def insert_layer(self, layer):
        if len(self.layers) > 0:
            self.layers[-1].out_flag = False
        self.layers.append(layer)
        self.layers[-1].out_flag = True

    def label_prediction(self, X):
        self.X_pred = X
        self.y_pred = []
        temp_val = X.shape[0]
        for i in range(temp_val):
            x = np.expand_dims(np.transpose(X[i, :]), axis=1)
            h = x.copy()
            for layer in self.layers:
                h = layer.forward_feed(h)
            self.y_pred.append(h)
        self.y_pred = np.array(self.y_pred, ndmin=2)
        return self.y_pred

    def train_model(self, X, y, val_data=None, batch_size=32, epochs=10):
        self.X = X
        self.y = y
        self.val_data = val_data
        self.batch_size = batch_size
        self.epochs = epochs
        self.trace_log = {'error': [], 'error_value': []}
        temp_val = X.shape[0]
        for epoch in range(self.epochs):
            tot_error = 0
            for i in range(temp_val):
                x = np.expand_dims(np.transpose(X[i, :]), axis=1)
                h = x.copy()
                for layer in self.layers:
                    h = layer.forward_feed(h)
                training_error = self.error(self.y[i], h)
                tot_error += training_error
                for layer in reversed(range(len(self.layers))):
                    if layer > 0:
                        if self.layers[layer].out_flag == True:
                            self.layers[layer].backward_propagation(
                                self.y[i], weights_data=None)
                        else:
                            self.layers[layer].backward_propagation(
                                self.layers[layer + 1].difference, weights_data=self.layers[layer + 1].wghts)
                        self.layers[layer].gradient += np.matmul(
                            self.layers[layer - 1].odata, np.transpose(self.layers[layer].difference))
            for layer in reversed(range(len(self.layers))):
                if layer > 0:
                    self.layers[layer].gradient /= temp_val
                    self.layers[layer].wghts -= self.lrate * self.layers[layer].gradient
            mean_error = tot_error / temp_val
            if self.val_data == None:
                pass
            else:
                val_X = self.val_data[0]
                val_y = self.val_data[1]
                tval_shape = val_X.shape[0]
                tval_error = 0
                for i in range(tval_shape):
                    val_x = np.expand_dims(np.transpose(val_X[i, :]), axis=1)
                    val_h = val_x.copy()
                    for layer in self.layers:
                        val_h = layer.forward_feed(val_h)
                    error_value = self.error(val_y[i], h)
                    tval_error += error_value
                mean_value_error = tval_error / tval_shape
                self.trace_log['error'].append(mean_error)
                self.trace_log['error_value'].append(mean_value_error)
        self.trace_log['error'] = np.squeeze(self.trace_log['error'])
        self.trace_log['error_value'] = np.squeeze(self.trace_log['error_value'])
        return self.trace_log

def binary_crossentropy(y, h):
    return y * (-np.log(h)) + (1 - y) * (-np.log(1 - h))

def relu(z):
    return z * (z > 0)

def relu_grad(z):
    return np.array(z > 0, dtype=np.int64)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def shuffle_split_data(X, y, percent):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, percent)
    X_train = X[split]
    y_train = y[split]
    X_test = X[~split]
    y_test = y[~split]
    return X_train, y_train, X_test, y_test

def extract_filedata(file_names):
    for file in file_names:
        if 'train_data' in file:
            train_data = np.genfromtxt(
                file, delimiter=',', missing_values='', filling_values=0.0)
        elif 'train_label' in file:
            train_label = np.genfromtxt(file, delimiter=',')
        elif 'test_data' in file:
            test_data = np.genfromtxt(
                file, delimiter=',', missing_values='', filling_values=0.0)
    return (test_data, train_data, train_label)

def write_output_file(output):
    np.savetxt('test_predictions.csv', output, fmt='%d', delimiter=',',newline='\n', header='', footer='', encoding=None)

if __name__ == "__main__":
    test_data, train_data, train_label = extract_filedata(sys.argv[1::])

    y = train_label
    X = train_data

    y = np.expand_dims(np.array(y), axis=1)

    X_train, y_train, X_test, y_test = shuffle_split_data(X, y, 70)
    X_train, y_train, X_val, y_val = shuffle_split_data(X_train, y_train, 20)

    X_train.shape
    y_train.shape

    mlp = MultiLayerPerceptron(error=binary_crossentropy, lrate=0.1)
    mlp.insert_layer(StartLayer(i_node_count=2))
    mlp.insert_layer(HiddenLayer(i_node_count=2, o_node_count=64,
                     actv=relu, actv_gradient=relu_grad))
    mlp.insert_layer(HiddenLayer(i_node_count=65, o_node_count=64,
                     actv=relu, actv_gradient=relu_grad))
    mlp.insert_layer(HiddenLayer(i_node_count=65, o_node_count=1,
                     actv=sigmoid, actv_gradient=sigmoid_grad))

    val_data = (X_val, y_val)

    batch_size = 32
    epochs = 200

    trace_log = mlp.train_model(
        X_train, y_train, val_data=val_data, batch_size=batch_size, epochs=epochs)

    np.argmin(trace_log['error_value'])
    epochs_range = range(epochs)
    np.argmin(trace_log['error_value'])

    y_pred = mlp.label_prediction(test_data)
    y_pred = np.array(y_pred > 0.5, dtype=np.int64)
    y_pred = np.squeeze(y_pred)

    write_output_file(y_pred)
