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
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(0.0, 0.1, (num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(0.0, 0.1, (num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_loss__dw_out = np.dot(delta_out.T, a_h)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_loss__a_h = np.dot(delta_out, self.weight_out)
        d_a_h__d_z_h = a_h * (1. - a_h)

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, x)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h


def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    m_w_h = np.zeros_like(model.weight_h)
    v_w_h = np.zeros_like(model.weight_h)
    m_b_h = np.zeros_like(model.bias_h)
    v_b_h = np.zeros_like(model.bias_h)

    m_w_out = np.zeros_like(model.weight_out)
    v_w_out = np.zeros_like(model.weight_out)
    m_b_out = np.zeros_like(model.bias_out)
    v_b_out = np.zeros_like(model.bias_out)

    t = 0

    for e in range(num_epochs):

        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:

            t += 1

            a_h, a_out = model.forward(X_train_mini)

            d_w_out, d_b_out, d_w_h, d_b_h = model.backward(
                X_train_mini, a_h, a_out, y_train_mini)

            m_w_h = beta1 * m_w_h + (1 - beta1) * d_w_h
            v_w_h = beta2 * v_w_h + (1 - beta2) * (d_w_h ** 2)
            model.weight_h -= learning_rate * (m_w_h / (1 - beta1**t)) / (np.sqrt(v_w_h / (1 - beta2**t)) + epsilon)

            m_b_h = beta1 * m_b_h + (1 - beta1) * d_b_h
            v_b_h = beta2 * v_b_h + (1 - beta2) * (d_b_h ** 2)
            model.bias_h -= learning_rate * (m_b_h / (1 - beta1**t)) / (np.sqrt(v_b_h / (1 - beta2**t)) + epsilon)

            m_w_out = beta1 * m_w_out + (1 - beta1) * d_w_out
            v_w_out = beta2 * v_w_out + (1 - beta2) * (d_w_out ** 2)
            model.weight_out -= learning_rate * (m_w_out / (1 - beta1**t)) / (np.sqrt(v_w_out / (1 - beta2**t)) + epsilon)

            m_b_out = beta1 * m_b_out + (1 - beta1) * d_b_out
            v_b_out = beta2 * v_b_out + (1 - beta2) * (d_b_out ** 2)
            model.bias_out -= learning_rate * (m_b_out / (1 - beta1**t)) / (np.sqrt(v_b_out / (1 - beta2**t)) + epsilon)

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc*100:.2f}% '
              f'| Valid Acc: {valid_acc*100:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc





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

def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.001):
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h =                 model.backward(X_train_mini, a_h, a_out, y_train_mini)

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        #### Epoch Logging ####        
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc




np.random.seed(123) # for the training set shuffling

epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid,
    num_epochs=50, learning_rate=0.001)


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
