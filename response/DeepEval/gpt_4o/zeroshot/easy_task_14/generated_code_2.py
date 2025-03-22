import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    input_shape = x_train.shape[1:]
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Define the model
    inputs = Input(shape=input_shape)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(inputs)
    
    # Fully connected layers for channel correlations
    fc1 = Dense(units=input_shape[2], activation='relu')(x)
    fc2 = Dense(units=input_shape[2], activation='sigmoid')(fc1)
    
    # Reshape to align with the input shape
    scale = Reshape((1, 1, input_shape[2]))(fc2)
    
    # Multiply the weights element-wise with the input feature map
    scaled_inputs = Multiply()([inputs, scale])
    
    # Flatten and final dense layer for classification
    x = Flatten()(scaled_inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()