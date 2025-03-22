import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    
    # Branch 2: 1x1, 3x3, 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(3, 3))(input_layer)
    
    # Concatenate feature maps from different branches
    concat = Concatenate()([branch1, branch2, branch3])
    
    # Flatten and pass through fully connected layers
    flat = Flatten()(concat)
    output = Dense(10, activation='softmax')(flat)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[SparseCategoricalAccuracy()])
    
    return model