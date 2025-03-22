import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    cifar10 = Cifar10()
    
    # Define the model architecture
    inputs = Input(shape=(32, 32, 3))
    x = inputs
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = Concatenate()(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # Block 2
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = Permute((3, 1, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # Block 3
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # Branch
    branch = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch = Conv2D(64, (3, 3), activation='relu')(branch)
    branch = Conv2D(64, (5, 5), activation='relu')(branch)
    branch = Concatenate()([branch, branch, branch])
    branch = MaxPooling2D(pool_size=(2, 2))(branch)
    branch = BatchNormalization()(branch)
    branch = Flatten()(branch)
    
    # Combine
    x = Concatenate()([x, branch])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=x)
    return model