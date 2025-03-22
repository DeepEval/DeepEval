import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Pooling layer 1 with 1x1 window and stride 1x1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv1)
    
    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    
    # Pooling layer 2 with 2x2 window and stride 2x2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    
    # Convolutional layer 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    
    # Pooling layer 3 with 4x4 window and stride 4x4
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(conv3)
    
    # Flatten the output and concatenate
    flat_pooled = Concatenate()(
        [pool1.output, pool2.output, pool3.output])
    
    # Fully connected layers
    dense1 = Flatten()(batch_normalization(flat_pooled))
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(64, activation='relu')(dense2)
    output_layer = Dense(10, activation='softmax')(dense3)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()