from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a 1x1 initial convolutional layer, three distinct branches, 
    a concatenation layer, a 1x1 convolutional layer, and a fully connected layer.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Local feature extraction branch
    x_local = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x_local = layers.Conv2D(32, (3, 3), activation='relu')(x_local)
    
    # Downsampling and upsampling branch 1
    x_down1 = layers.AveragePooling2D((2, 2))(x)
    x_down1 = layers.Conv2D(32, (3, 3), activation='relu')(x_down1)
    x_up1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu')(x_down1)
    
    # Downsampling and upsampling branch 2
    x_down2 = layers.AveragePooling2D((2, 2))(x)
    x_down2 = layers.Conv2D(32, (3, 3), activation='relu')(x_down2)
    x_up2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu')(x_down2)
    
    # Concatenate the outputs of the branches
    x_concat = layers.Concatenate()([x_local, x_up1, x_up2])
    
    # Refine the output using a 1x1 convolutional layer
    x_refine = layers.Conv2D(64, (1, 1), activation='relu')(x_concat)
    
    # Flatten the output to pass it to the fully connected layer
    x_flatten = layers.Flatten()(x_refine)
    
    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x_flatten)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create an instance of the model
model = dl_model()
model.summary()