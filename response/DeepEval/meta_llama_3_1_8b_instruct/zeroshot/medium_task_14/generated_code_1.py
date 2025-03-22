# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of three sequential blocks, each comprising a convolutional layer, a batch normalization layer, and a ReLU activation function to extract image features.
    These blocks produce three separate output paths, each corresponding to one block's output.
    Additionally, a parallel branch of convolutional layers processes the input directly.
    The outputs from all paths are added, and the aggregated result passes through two fully connected layers for classification.
    """
    
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the model architecture
    inputs = keras.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    block1_output = layers.MaxPooling2D(2, 2)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, activation='relu')(block1_output)
    x = layers.BatchNormalization()(x)
    block2_output = layers.MaxPooling2D(2, 2)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, activation='relu')(block2_output)
    x = layers.BatchNormalization()(x)
    block3_output = layers.MaxPooling2D(2, 2)(x)
    
    # Parallel branch
    x = layers.Conv2D(64, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    # Add the outputs from all paths
    x = layers.Concatenate()([block1_output, block2_output, block3_output, x])
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    
    return model