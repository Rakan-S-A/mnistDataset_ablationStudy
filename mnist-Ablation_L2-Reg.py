import time
import sys
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

sys.path.insert(0, '..')

# # Chapter 11 - Implementing a Multi-layer Artificial Neural Network from Scratch
# 

print('Fetching dataset...')

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print('Starting training...')

# --- START TIMER ---
start_time = time.time()

# Normalize to [-1, 1] range:



X = ((X / 255.) - .5) * 2


# Split into training, validation, and test set:





X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)


# optional to free up some memory by deleting non-used arrays:
del X_temp, y_temp, X, y



# ## Implementing a multi-layer perceptron







##########################
### MODEL
##########################

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        # hidden
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y, l2_lambda=0.0):  
        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out) + 2 * l2_lambda * self.weight_out
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h) + 2 * l2_lambda * self.weight_h
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h





model = NeuralNetMLP(num_features=28*28,
                     num_hidden=100,
                     num_classes=10)


# ## Coding the neural network training loop

# Defining data loaders:



#Initial validation
num_epochs = 1
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]

        
# iterate over training epochs
for i in range(num_epochs):

    # iterate over minibatches
    minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
    
    for X_train_mini, y_train_mini in minibatch_gen:

        break
        
    break
    
print(X_train_mini.shape)
print(y_train_mini.shape)


# Defining a function to compute the loss and accuracy



def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets) 


_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)

print(f'Initial validation MSE: {mse:.1f}')
print(f'Initial validation accuracy: {acc*100:.1f}%')



def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
        
    for i, (features, targets) in enumerate(minibatch_gen):

        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/(i+1)
    acc = correct_pred/num_examples
    return mse, acc




mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')



#############################################################

np.random.seed(123)
num_epochs = 50
minibatch_size = 100
learning_rate = 0.01
rho = 0.9
epsilon = 1e-8
l2_lambda = 0.000001

model = NeuralNetMLP(num_features=28*28, num_hidden=100, num_classes=10)

# RMSProp caches outside class
cache_w_h = np.zeros_like(model.weight_h)
cache_b_h = np.zeros_like(model.bias_h)
cache_w_out = np.zeros_like(model.weight_out)
cache_b_out = np.zeros_like(model.bias_out)

epoch_loss, epoch_train_acc, epoch_valid_acc = [], [], []

for e in range(num_epochs):
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_mini, y_mini in minibatch_gen:
        # Forward pass
        a_h, a_out = model.forward(X_mini)
        # Backward pass
        d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_mini, a_h, a_out, y_mini, l2_lambda=l2_lambda)

        
        # RMSProp cache update
        cache_w_h = rho*cache_w_h + (1-rho)*d_loss__d_w_h**2
        cache_b_h = rho*cache_b_h + (1-rho)*d_loss__d_b_h**2
        cache_w_out = rho*cache_w_out + (1-rho)*d_loss__dw_out**2
        cache_b_out = rho*cache_b_out + (1-rho)*d_loss__db_out**2
        
        # Parameter update
        model.weight_h -= learning_rate * d_loss__d_w_h / (np.sqrt(cache_w_h) + epsilon)
        model.bias_h -= learning_rate * d_loss__d_b_h / (np.sqrt(cache_b_h) + epsilon)
        model.weight_out -= learning_rate * d_loss__dw_out / (np.sqrt(cache_w_out) + epsilon)
        model.bias_out -= learning_rate * d_loss__db_out / (np.sqrt(cache_b_out) + epsilon)
        
    # Epoch metrics
    train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
    valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
    epoch_loss.append(train_mse)
    epoch_train_acc.append(train_acc*100)
    epoch_valid_acc.append(valid_acc*100)
    print(f"Epoch {e+1:03d}/{num_epochs:03d} | Train MSE: {train_mse:.4f} | Train Acc: {train_acc*100:.2f}% | Valid Acc: {valid_acc*100:.2f}%")




np.random.seed(123) # for the training set shuffling

#epoch_loss, epoch_train_acc, epoch_valid_acc = train(
   # model, X_train, y_train, X_valid, y_valid,
   # num_epochs=50, learning_rate=0.1)


# ## Evaluating the neural network performance


# --- END TIMER ---
end_time = time.time()
    
    # Calculate duration
elapsed_time = end_time - start_time
print(f"\nTraining complete! Time taken: {elapsed_time:.2f} seconds")

'''
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
#plt.savefig('figures/11_07.png', dpi=300)
plt.show()
'''


'''
plt.plot(range(len(epoch_train_acc)), epoch_train_acc,
         label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc,
         label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
#plt.savefig('figures/11_08.png', dpi=300)
plt.show()
'''



test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

