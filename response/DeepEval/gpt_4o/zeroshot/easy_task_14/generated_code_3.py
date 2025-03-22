from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load the CIFAR-10 dataset to get input shape
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]  # (32, 32, 3)
    
    # Define the input
    inputs = Input(shape=input_shape)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(inputs)
    
    # First fully connected layer to generate weights
    x = Dense(units=input_shape[-1], activation='relu')(x)
    
    # Second fully connected layer
    x = Dense(units=input_shape[-1], activation='sigmoid')(x)
    
    # Reshape to align with input channels
    x = Reshape((1, 1, input_shape[-1]))(x)
    
    # Multiply element-wise with the input feature map
    x = Multiply()([inputs, x])
    
    # Flatten the result
    x = Flatten()(x)
    
    # Final fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
model = dl_model()
model.summary()