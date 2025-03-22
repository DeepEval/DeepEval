import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LayerNormalization, Add, Dense, Input

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Define the functional model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Spatial feature extraction with depthwise separable convolution
    x = Conv2D(32, (7, 7), padding='same', activation='relu')(input_layer)
    x = DepthwiseConv2D((7, 7), activation='relu')(x)
    x = LayerNormalization()(x)
    
    # Flatten and pass through two fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    # Add processed features with original input
    combined = Add()([x, input_layer])
    
    # Output layer for classification
    output_layer = Dense(10, activation='softmax')(combined)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()