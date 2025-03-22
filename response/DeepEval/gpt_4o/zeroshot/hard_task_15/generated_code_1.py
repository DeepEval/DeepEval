from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 data for input shape
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Define input shape
    input_shape = x_train.shape[1:]  # CIFAR-10 images are 32x32x3
    
    # Main Input
    inputs = Input(shape=input_shape)
    
    # Main path
    # Global Average Pooling
    x = GlobalAveragePooling2D()(inputs)
    # Two Fully Connected Layers to generate weights
    x = Dense(128, activation='relu')(x)
    x = Dense(input_shape[-1], activation='sigmoid')(x)  # Generate weights with same channels
    
    # Reshape weights to match input layer's shape (1, 1, channels)
    x = Dense(input_shape[-1], activation='sigmoid')(x)
    
    # Element-wise multiplication with input feature map
    x = Multiply()([inputs, x])

    # Branch path: direct connection to input layer
    branch = inputs
    
    # Add outputs of main and branch paths
    combined = Add()([x, branch])
    
    # Two Fully Connected Layers
    x = Flatten()(combined)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes
    
    # Construct model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()