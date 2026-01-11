import numpy as np
import nnfs
import math
import plotext as plt
import requests
from NNdebug_logger import DebugLogger

def load_diabetes_data():
    # Load CSV data
    data = np.loadtxt('data/diabetic_data.csv', delimiter=',', skiprows=1, dtype=str)
    
    # Select features and target
    feature_indices = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 46, 47, 48]
    target_index = 49

    X_raw = data[:, feature_indices]
    y_raw = data[:, target_index]
    
    X = []

    # Define mappings for categorical features
    race_map = {'Caucasian': [1,0,0,0], 'AfricanAmerican': [0,1,0,0], 'Other':[0,0,1,0], '?':[0,0,0,1]}
    gender_map = {'Female':[1,0], 'Male':[0,1]}
    age_map = {'[0-10)':[1,0,0,0,0,0,0,0,0,0],
               '[10-20)':[0,1,0,0,0,0,0,0,0,0],
               '[20-30)':[0,0,1,0,0,0,0,0,0,0],
               '[30-40)':[0,0,0,1,0,0,0,0,0,0],
               '[40-50)':[0,0,0,0,1,0,0,0,0,0],
               '[50-60)':[0,0,0,0,0,1,0,0,0,0],
               '[60-70)':[0,0,0,0,0,0,1,0,0,0],
               '[70-80)':[0,0,0,0,0,0,0,1,0,0],
               '[80-90)':[0,0,0,0,0,0,0,0,1,0],
               '[90-100)':[0,0,0,0,0,0,0,0,0,1]}
    specialty_map = {'?':[1,0,0,0,0,0], 
                     'InternalMedicine':[0,1,0,0,0,0],
                     'Family/GeneralPractice':[0,0,1,0,0,0],
                     'Cardiology':[0,0,0,1,0,0],
                     'Surgery-General':[0,0,0,0,1,0],
                     'Pediatrics-Endocrinology':[0,0,0,0,0,1]}
    glu_map = {'None':[1,0,0,0], '>200':[0,1,0,0], '>300':[0,0,1,0], 'Norm':[0,0,0,1]}
    a1c_map = {'None':[1,0,0,0], '>7':[0,1,0,0], '>8':[0,0,1,0], 'Norm':[0,0,0,1]}
    change_map = {'No':[1,0], 'Ch':[0,1]}
    med_map = {'No':[1,0], 'Yes':[0,1]}

    # Features processing
    for row in X_raw:
        features = []
        
        # One-hot categorical features
        features.extend(race_map.get(row[0], [0,0,0,1]))
        features.extend(gender_map.get(row[1], [0,0]))
        features.extend(age_map.get(row[2], [0,0,0,0,0,1,0,0,0,0]))
        
        # Numerical features: admission_type_id, discharge_disposition_id, admission_source_id
        for j in [3,4,5]:
            try:
                features.append(float(row[j]))
            except:
                features.append(0)
        
        # time_in_hospital
        try:
            features.append(float(row[6]))
        except:
            features.append(3)
        
        # medical_specialty one-hot
        features.extend(specialty_map.get(row[7], [1,0,0,0,0,0]))
        
        # Numerical features: num_lab_procedures, num_procedures, num_medications, etc.
        for j in [8,9,10,11,12,13,14,15]:
            try:
                features.append(float(row[j]))
            except:
                features.append(0)
        
        # Categorical one-hot
        features.extend(glu_map.get(row[16], [1,0,0,0]))
        features.extend(a1c_map.get(row[17], [1,0,0,0]))
        features.extend(change_map.get(row[18], [1,0]))
        features.extend(med_map.get(row[19], [1,0]))

        X.append(features)
    
    X = np.array(X, dtype=float)
    
    # Process target
    y = []
    for target in y_raw:
        if target == 'NO':
            y.append(0)
        elif target == '<30':
            y.append(1)
        else:
            y.append(2)
    y = np.array(y, dtype='uint8')
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Normalize numerical columns (original numerical indices shifted due to one-hot)
    # Let’s find numerical indices:
    # Original: [3,4,5,6,8,9,10,11,12,13,14,15]
    # One-hot additions: race(4)+gender(2)+age(10)+specialty(6)=22 extra columns before numerical
    numerical_indices_shifted = [22+0, 22+1, 22+2, 22+3]  # admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital
    numerical_indices_shifted += list(range(22+4+6, 22+4+6+8))  # num_lab..num_medications

    for idx in numerical_indices_shifted:
        mean_val = np.mean(X[:, idx])
        std_val = np.std(X[:, idx]) + 1e-8
        X[:, idx] = (X[:, idx] - mean_val) / (3 * std_val)

    # Split
    n_train = int(len(X) * 2/3)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)

    print(f"Training: {len(X_train)} samples, {X_train.shape[1]} features (one-hot applied)")
    print(f"Testing: {len(X_test)} samples")
    print(f"Target distribution - Train: {np.bincount(y_train)}")
    print(f"Target distribution - Test: {np.bincount(y_test)}")
    print(f"Class weights: {class_weights}")
    
    return X_train, y_train, X_test, y_test, class_weights

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, alpha=0.01):
        he_std = np.sqrt(2.0 / ((1 + alpha**2) * n_inputs))
        self.weights = np.random.randn(n_inputs, n_neurons) * he_std
        self.biases = np.zeros((1, n_neurons))

        # Adam optimizer state
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.t = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, inputs, dL_dout):
        batch_size = inputs.shape[0]

        self.dweights = np.dot(inputs.T, dL_dout) / batch_size
        self.dbiases = np.sum(dL_dout, axis=0, keepdims=True) / batch_size

        # Gradient clipping
        grad_norm = np.linalg.norm(self.dweights)
        if grad_norm > 1.0:
            self.dweights /= grad_norm
            self.dbiases /= grad_norm

        return np.dot(dL_dout, self.weights.T)

    def update_adam(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.dweights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (self.dweights ** 2)

        self.m_biases = beta1 * self.m_biases + (1 - beta1) * self.dbiases
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (self.dbiases ** 2)

        m_w_corr = self.m_weights / (1 - beta1 ** self.t)
        v_w_corr = self.v_weights / (1 - beta2 ** self.t)

        m_b_corr = self.m_biases / (1 - beta1 ** self.t)
        v_b_corr = self.v_biases / (1 - beta2 ** self.t)

        self.weights -= learning_rate * m_w_corr / (np.sqrt(v_w_corr) + epsilon)
        self.biases -= learning_rate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)



class BatchNorm:
    def __init__(self, n_features, momentum=0.9, eps=1e-8):
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))

        self.momentum = momentum
        self.eps = eps

        # Running statistics (used during inference)
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

    def forward(self, x, training=True):
        self.x = x

        if training:
            # Batch statistics
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * self.batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * self.batch_var
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            # Inference mode
            mean = self.running_mean
            var = self.running_var

        # Normalize
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv

        # Scale and shift
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        N = dout.shape[0]

        # Gradients for scale and shift
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradient w.r.t normalized input
        dxhat = dout * self.gamma

        # Backprop through normalization
        dvar = np.sum(
            dxhat * self.x_centered * -0.5 * self.std_inv**3,
            axis=0,
            keepdims=True
        )

        dmean = (
            np.sum(dxhat * -self.std_inv, axis=0, keepdims=True)
            + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)
        )

        dx = (
            dxhat * self.std_inv
            + dvar * 2.0 * self.x_centered / N
            + dmean / N
        )

        return dx


class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, inputs):
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
    def backward(self, output, dL_dOutput):
        gradient = np.where(output > 0, dL_dOutput, self.alpha * dL_dOutput)
        return gradient
    
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    def backward(self, n_inputs, dL_dOutput):
        gradient = dL_dOutput.copy()
        gradient[n_inputs <= 0] = 0
        return gradient # return the gradient in terms of L, where it is 0 if less than or equal to 0


#Softmax Activation is meant to reduce all numbers into probabilites
class Activation_Softmax:
    def forward(self, inputs):
        # Clip inputs to prevent overflow
        inputs_clipped = np.clip(inputs, -500, 500)
        exp_values = np.exp(inputs_clipped - np.max(inputs_clipped, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)
    def backwardPlus(self, softmaxOut, y_true):
        if(y_true.ndim == 1):
            y_true = np.eye(softmaxOut.shape[1])[y_true]
        gradient = softmaxOut - y_true
        return gradient



class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    def gradient(self, y_pred, y_true):
        partial_derivatives = self.backward(y_pred, y_true)

class Loss_CategoricalCrossEntropy(Loss):
    def __init__(self, class_weights=None):
        self.class_weights = class_weights
        
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if len(y_true.shape) == 1:
                weights = self.class_weights[y_true]
            else:
                weights = np.sum(y_true * self.class_weights, axis=1)
            negative_log_likelihoods *= weights
            
        return negative_log_likelihoods
    def accuracy(self, y_pred, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy
    #there is no point in finding doing Loss backwards, since we can combine Loss and softmax into one stap
    def backward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:#1d scalar values
            #using y_pred, since the size has been compressed to 1 in y_true if it's 1D
            y_true = np.eye(y_pred.shape[1])[y_true]
        #now we have a 2d matric of 1 hot rows
        gradient = -y_true/y_pred #the partial derivative formula is -1/x
        return gradient
        

def log_debug_info(logger, epoch, loss, acc, learning_rate, dense1, dense2, dense3):
    logger.log_epoch(epoch, loss, acc, learning_rate)
    if epoch > 0:  # Skip gradients on first epoch
        logger.log_gradients(epoch, dense1, dense2, dense3)
    logger.log_weights(epoch, dense1, dense2, dense3)
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

def test_accuracy(dense1, activation1, dense2, dense3, activation2, X_test, y_test, n_samples=500):
    # Randomly sample test data
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Forward pass on test data
    dense1.forward(X_sample)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    dense3.forward(dense2.output)
    activation2.forward(dense3.output)
    
    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y_sample)
    
    print(f"Test Accuracy on {len(X_sample)} samples: {accuracy:.4f}")
    return accuracy
    # Plot age vs glucose level colored by stroke outcome
    no_stroke = X[y == 0]
    stroke = X[y == 1]
    
    if len(no_stroke) > 0:
        plt.scatter(no_stroke[:, 0], no_stroke[:, 3], marker="o", label="No Stroke")
    if len(stroke) > 0:
        plt.scatter(stroke[:, 0], stroke[:, 3], marker="x", label="Stroke")
    
    plt.title("Stroke Data (Age vs Glucose Level)")
    plt.xlabel("Age (normalized)")
    plt.ylabel("Glucose Level (normalized)")
    plt.show()

def trainingData():
    logger = DebugLogger("diabetes_debug.log")
    X_train, y_train, X_test, y_test, class_weights = load_diabetes_data()
    
    # Optimized hyperparameters
    initial_lr = 0.0015
    batch_size = 256  # Larger batch for efficiency
    epochs = 3000
    
    # Larger network with batch normalization
    dense1 = Layer_Dense(X_train.shape[1], 64)
    bn1 = BatchNorm(64)
    activation1 = Activation_LeakyReLU()
    dense2 = Layer_Dense(64, 32)
    bn2 = BatchNorm(32)
    activation2 = Activation_LeakyReLU()
    dense3 = Layer_Dense(32, 3)
    activation3 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossEntropy(class_weights)
    
    for epoch in range(epochs):
        learning_rate = initial_lr / (1 + 0.0005 * epoch)
        
        # Shuffle training data
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Mini-batch training
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            dense1.forward(X_batch)
            bn1_out = bn1.forward(dense1.output, training=True)
            activation1.forward(bn1_out)
            dense2.forward(activation1.output)
            bn2_out = bn2.forward(dense2.output, training=True)
            activation2.forward(bn2_out)
            dense3.forward(activation2.output)
            activation3.forward(dense3.output)
            
            # Backward pass with batch normalization
            dL_dz = activation3.backwardPlus(activation3.output, y_batch)
            dL_ddense3 = dense3.backward(activation2.output, dL_dz)
            dL_dactivation2 = activation2.backward(bn2_out, dL_ddense3)
            dL_dbn2 = bn2.backward(dL_dactivation2)
            dL_ddense2 = dense2.backward(activation1.output, dL_dbn2)
            dL_dactivation1 = activation1.backward(bn1_out, dL_ddense2)
            dL_dbn1 = bn1.backward(dL_dactivation1)
            dense1.backward(X_batch, dL_dbn1)
            
            # Update weights with Adam
            dense3.update_adam(learning_rate)
            dense2.update_adam(learning_rate)
            dense1.update_adam(learning_rate)
        
        # Log every 500 epochs
        if epoch % 500 == 0:
            # Sample-based metrics to avoid full dataset forward pass
            sample_indices = np.random.choice(len(X_train), 1000, replace=False)
            X_sample = X_train[sample_indices]
            y_sample = y_train[sample_indices]
            
            dense1.forward(X_sample)
            bn1_out = bn1.forward(dense1.output, training=False)
            activation1.forward(bn1_out)
            dense2.forward(activation1.output)
            bn2_out = bn2.forward(dense2.output, training=False)
            activation2.forward(bn2_out)
            dense3.forward(activation2.output)
            activation3.forward(dense3.output)
            
            loss = loss_function.calculate(activation3.output, y_sample)
            acc = loss_function.accuracy(activation3.output, y_sample)
            
            log_debug_info(logger, epoch, loss, acc, learning_rate, dense1, dense2, dense3)
            test_accuracy(dense1, activation1, dense2, dense3, activation3, X_test, y_test, 500)
    
    print("\nFinal Test Results:")
    test_accuracy(dense1, activation1, dense2, dense3, activation3, X_test, y_test, 1000)
    return    
    # Log every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Accuracy: {acc:.4f}")
        # Test on random sample every 500 epochs
        test_accuracy(dense1, activation1, dense2, dense3, activation2, X_test, y_test, 500)
    
        # Backward pass
        dL_dz = activation2.backwardPlus(activation2.output, y_train)
        dL_ddense3 = dense3.backward(dense2.output, dL_dz)
        dense3.update(learning_rate)
        dL_ddense2 = dense2.backward(activation1.output, dL_ddense3)
        dense2.update(learning_rate)
        dL_dreluOut = activation1.backward(dense1.output, dL_ddense2)
        dense1.backward(X_train, dL_dreluOut)
        dense1.update(learning_rate)
    
    # Final test on larger sample
    print("\nFinal Test Results:")
    test_accuracy(dense1, activation1, dense2, dense3, activation2, X_test, y_test, 1000)

    


# Run the neural network
trainingData()
