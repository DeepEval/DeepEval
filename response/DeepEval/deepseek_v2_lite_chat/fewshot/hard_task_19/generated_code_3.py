import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Reshape, multiply

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to [-1, 1]
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

# Number of classes
num_classes = 10

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
        global_pool = AveragePooling2D(pool_size=(4, 4), padding='valid')(conv3)
        
        # Branch path
        flatten = Flatten()(global_pool)
        fc1 = Dense(units=512, activation='relu')(flatten)
        channel_weights = Dense(units=3, activation='sigmoid')(fc1)
        reshaped_weights = Reshape((3, 1, 1))(channel_weights)
        channel_weighted_input = multiply([global_pool, reshaped_weights])
        
        return input_layer, channel_weighted_input
    
    # Forward pass to get outputs from both paths
    input_layer, channel_weighted_input = main_path(input_layer)
    
    # Fully connected layers for classification
    fc1 = Dense(units=1024, activation='relu')(channel_weighted_input)
    fc2 = Dense(units=512, activation='relu')(fc1)
    output_layer = Dense(units=num_classes, activation='softmax')(fc2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)