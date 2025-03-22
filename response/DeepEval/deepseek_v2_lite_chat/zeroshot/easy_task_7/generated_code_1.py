import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Define the main path of the model
def main_path(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    return Model(inputs=input_layer.input, outputs=x, name='main_path')

# Define the branch path of the model
def branch_path(input_shape):
    input_branch = Input(shape=input_shape)
    output_branch = Dense(10, activation='softmax')(input_branch)
    return Model(inputs=input_branch.input, outputs=output_branch, name='branch_path')

# Define the combined model
def dl_model():
    input_shape = (28, 28, 1)  # MNIST image shape
    
    main_path_output = main_path(input_shape)(x_train)
    branch_path_output = branch_path(input_shape)(x_train)
    
    combined_output = Concatenate()([main_path_output, branch_path_output])
    output = Flatten()(combined_output)
    output = Dense(10, activation='softmax')(output)  # Softmax for multi-class classification
    
    model = Model(inputs=[main_path_output.input, branch_path_output.input], outputs=output, name='main_and_branch_model')
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# Build and return the model
model = dl_model()
model.summary()