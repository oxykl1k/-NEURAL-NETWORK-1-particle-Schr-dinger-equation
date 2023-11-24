# Importing the necessary libraries
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

# Define the neural network architecture
tf.keras.backend.set_floatx('float64')
class SchrodingerNet(tf.keras.Model):
    def __init__(self):
        super(SchrodingerNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Create an instance of the SchrodingerNet model
model = SchrodingerNet()

# Define the loss function (MSE) for the neural network
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Generate example data for training
x_train = np.linspace(-5, 5, 1000).reshape(-1, 1).astype(np.float64)
y_train = np.sqrt(2/np.pi) * np.exp(-np.square(x_train)) * np.sin(x_train)

# Perform training
for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss_value = loss_fn(y_train, y_pred)
    
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

# Generate test data
x_test = np.linspace(-10, 10, 1000).reshape(-1, 1).astype(np.float64)

# Predict the wave function using the trained model
y_pred_test = model(x_test)

# Plot the results
plt.plot(x_test, np.abs(y_pred_test) ** 2)
plt.xlabel('Position (x)')
plt.ylabel('Probability Density')
plt.title('Wave Function')
plt.show()