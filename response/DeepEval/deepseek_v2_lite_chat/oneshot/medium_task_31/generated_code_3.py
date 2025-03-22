import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10
import tensorflow as tf

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the convolutional layers for each group
    def conv_block(input_tensor, filters, kernel_size):
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
    
    conv1 = conv_block(split1, 32, (1, 1))
    conv2 = conv_block(split2, 32, (3, 3))
    conv3 = conv_block(split3, 32, (5, 5))
    
    # Concatenate the outputs from the three groups
    fused_features = Concatenate()([conv1, conv2, conv3])
    
    # Batch normalization and flattening
    bn_layer = BatchNormalization()(fused_features)
    flattened = Flatten()(bn_layer)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)