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

#1 f
class NeuralNetMLP3:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        # Hidden layer 1
        self.w1 = rng.normal(0.0, 0.1, (num_hidden, num_features))
        self.b1 = np.zeros(num_hidden)

        # Hidden layer 2
        self.w2 = rng.normal(0.0, 0.1, (num_hidden, num_hidden))
        self.b2 = np.zeros(num_hidden)

        # Hidden layer 3
        self.w3 = rng.normal(0.0, 0.1, (num_hidden, num_hidden))
        self.b3 = np.zeros(num_hidden)

        # Output layer
        self.w_out = rng.normal(0.0, 0.1, (num_classes, num_hidden))
        self.b_out = np.zeros(num_classes)

    def forward(self, x):
        z1 = np.dot(x, self.w1.T) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, self.w2.T) + self.b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.w3.T) + self.b3
        a3 = sigmoid(z3)

        z_out = np.dot(a3, self.w_out.T) + self.b_out
        a_out = sigmoid(z_out)

        return (a1, a2, a3), a_out

    def backward(self, x, activations, a_out, y):
        a1, a2, a3 = activations
        y_onehot = int_to_onehot(y, self.num_classes)

        # Output layer delta
        delta_out = (2.0 * (a_out - y_onehot) / y.shape[0]) * a_out * (1.0 - a_out)

        dw_out = np.dot(delta_out.T, a3)
        db_out = np.sum(delta_out, axis=0)

        # Hidden layers deltas
        delta3 = np.dot(delta_out, self.w_out) * a3 * (1.0 - a3)
        delta2 = np.dot(delta3, self.w3) * a2 * (1.0 - a2)
        delta1 = np.dot(delta2, self.w2) * a1 * (1.0 - a1)

        # Gradients
        dw3 = np.dot(delta3.T, a2)
        db3 = np.sum(delta3, axis=0)

        dw2 = np.dot(delta2.T, a1)
        db2 = np.sum(delta2, axis=0)

        dw1 = np.dot(delta1.T, x)
        db1 = np.sum(delta1, axis=0)

        return dw_out, db_out, dw3, db3, dw2, db2, dw1, db1




#2 f

model = NeuralNetMLP3(
    num_features=28*28,
    num_hidden=100,
    num_classes=10
)



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
#3 f
def train(model, X_train, y_train, X_valid, y_valid,
          num_epochs, learning_rate=0.1):

    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):

        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size
        )

        for X_train_mini, y_train_mini in minibatch_gen:

            # -------- Forward pass --------
            activations, a_out = model.forward(X_train_mini)

            # -------- Backward pass --------
            (dw_out, db_out,
             dw3, db3,
             dw2, db2,
             dw1, db1) = model.backward(
                X_train_mini,
                activations,
                a_out,
                y_train_mini
            )

            # -------- Update weights --------
            model.w_out -= learning_rate * dw_out
            model.b_out -= learning_rate * db_out

            model.w3 -= learning_rate * dw3
            model.b3 -= learning_rate * db3

            model.w2 -= learning_rate * dw2
            model.b2 -= learning_rate * db2

            model.w1 -= learning_rate * dw1
            model.b1 -= learning_rate * db1

        # -------- Epoch evaluation --------
        train_mse, train_acc = compute_mse_and_acc(
            model, X_train, y_train
        )
        valid_mse, valid_acc = compute_mse_and_acc(
            model, X_valid, y_valid
        )

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        print(f'Epoch {e+1:03d}/{num_epochs:03d} | '
              f'Train MSE: {train_mse:.4f} | '
              f'Train Acc: {train_acc*100:.2f}% | '
              f'Valid Acc: {valid_acc*100:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc





np.random.seed(123) # for the training set shuffling

epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid,
    num_epochs=50, learning_rate=0.1)


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

