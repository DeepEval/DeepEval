import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Flatten, Reshape

# Ensure reproducibility
import numpy as np
np.random.seed(1)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch
    branch1 = GlobalAveragePooling2D()(input_layer)
    branch1 = Dense(1024, activation='relu')(branch1)
    branch1 = Dense(512, activation='relu')(branch1)
    reshaped_branch1 = Reshape((512, 2, 1))(branch1)
    
    # Second branch
    branch2 = GlobalAveragePooling2D()(input_layer)
    branch2 = Dense(1024, activation='relu')(branch2)
    branch2 = Dense(512, activation='relu')(branch2)
    reshaped_branch2 = Reshape((512, 2, 1))(branch2)
    
    # Concatenate and reshape
    concatenated = Concatenate(axis=-1)([reshaped_branch1, reshaped_branch2])
    reshaped_concat = Reshape((512, 2))(concatenated)
    
    # Final layer
    output_layer = Dense(10, activation='softmax')(reshaped_concat)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()


model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=100, validation_data=(x_test, y_test))