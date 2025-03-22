import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the first block with depthwise separable convolutional layers
    def first_block(x):
        x1 = layers.Lambda(lambda x: tf.split(x, 3, -1))(x)
        x1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x1[0])
        x1 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x1[0])
        x1 = layers.SeparableConv2D(32, (5, 5), activation='relu')(x1[0])
        
        x2 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x1[1])
        x2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x1[1])
        x2 = layers.SeparableConv2D(32, (5, 5), activation='relu')(x1[1])
        
        x3 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x1[2])
        x3 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x1[2])
        x3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(x1[2])
        
        x = layers.Concatenate()([x1, x2, x3])
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    # Define the second block with multiple branches for feature extraction
    def second_block(x):
        x1 = layers.Conv2D(32, (1, 1), activation='relu')(x)
        x1 = layers.Conv2D(32, (3, 3), activation='relu')(x1)
        
        x2 = layers.Conv2D(32, (1, 1), activation='relu')(x)
        x2 = layers.Conv2D(32, (3, 3), activation='relu')(x2)
        
        x3 = layers.MaxPooling2D((2, 2))(x)
        x3 = layers.Conv2D(32, (1, 1), activation='relu')(x3)
        
        x4 = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x4 = layers.Conv2D(32, (3, 3), activation='relu')(x4)
        
        x = layers.Concatenate()([x1, x2, x3, x4])
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    # Construct the deep learning model
    inputs = keras.Input(shape=(32, 32, 3))
    x = first_block(inputs)
    x = second_block(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Test the model
model = dl_model()
model.summary()