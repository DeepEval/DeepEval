import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, Reshape, Flatten, Dense, Add, Lambda
from keras.layers import DepthwiseConv2D, PReLU

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = DepthwiseConv2D(kernel_size=(7, 7))(inputs)
    x = LayerNormalization()(x)
    
    x = Conv2D(16, (1, 1), activation='relu')(x)
    x = LayerNormalization()(x)
    
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = LayerNormalization()(x)
    
    # Branch path
    branch_output = inputs
    
    # Combine the outputs of both paths
    combined_output = Add()([x, branch_output])
    
    # Flatten and process through two fully connected layers
    x = Flatten()(combined_output)
    
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model (this is a simplified example; you should add training loop)
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Save the model
model.save('cifar10_model.h5')