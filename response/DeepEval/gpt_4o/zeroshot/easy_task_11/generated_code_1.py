import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for MNIST images (28x28 grayscale images)
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling layer with 5x5 window and 3x3 stride
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)
    
    # 1x1 Convolutional layer to enhance depth and introduce nonlinearity
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the feature maps to a 1D vector
    x = Flatten()(x)
    
    # First fully connected layer
    x = Dense(units=128, activation='relu')(x)
    
    # Dropout layer to mitigate overfitting
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer
    x = Dense(units=64, activation='relu')(x)
    
    # Output layer with 10 units (one for each class) and softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with appropriate loss and optimizer for classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model