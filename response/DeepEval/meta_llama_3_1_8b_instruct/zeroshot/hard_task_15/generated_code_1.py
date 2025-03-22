# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. 
    In the main path, the input features undergo global average pooling to extract global information from the feature map.
    The pooled features are then passed through two fully connected layers to generate weights with the same number of channels as the input layer.
    These weights are reshaped to match the input layer's shape and multiplied element-wise with the input feature map.
    
    The branch path is directly connected to the input layer without any modification.
    The outputs from both the main and branch paths are added together.
    The combined result is then fed through two fully connected layers to produce the final probability distribution.
    """
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Define input shape
    input_shape = (32, 32, 3)
    
    # Create input layer
    inputs = keras.Input(shape=input_shape)
    
    # Main path
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Generate weights with the same number of channels as the input layer
    weights = layers.Dense(128 * 16 * 16)(x)
    weights = layers.Reshape((16, 16, 128))(weights)
    
    # Element-wise multiplication with input feature map
    x = layers.Multiply()([inputs, weights])
    
    # Branch path
    x_branch = layers.Conv2D(32, 3, activation='relu')(inputs)
    x_branch = layers.Conv2D(64, 3, activation='relu')(x_branch)
    x_branch = layers.Conv2D(128, 3, activation='relu')(x_branch)
    
    # Add outputs from both the main and branch paths
    x = layers.Add()([x, x_branch])
    
    # Flatten input for dense layers
    x = layers.Flatten()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()
print(model.summary())