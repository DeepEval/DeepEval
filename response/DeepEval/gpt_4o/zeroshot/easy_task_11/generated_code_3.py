from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import mnist

def dl_model():
    # Define the input shape for the MNIST dataset (28x28 images with a single channel)
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Average pooling layer with a 5x5 window and a 3x3 stride
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inputs)
    
    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # First fully connected layer
    x = Dense(units=128, activation='relu')(x)
    
    # Dropout layer to mitigate overfitting
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer
    x = Dense(units=64, activation='relu')(x)
    
    # Output layer for 10 classes with softmax activation for probability distribution
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with categorical crossentropy loss and an optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model