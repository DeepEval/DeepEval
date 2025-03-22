import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Define the main path
def main_path(input_shape):
    input_layer = Input(shape=input_shape)
    conv = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    avg_pool = GlobalAveragePooling2D()(pool)
    
    dense1 = Dense(256, activation='relu')(avg_pool)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Define the branch path
def branch_path(input_shape):
    input_layer = Input(shape=input_shape)
    conv = Conv2D(32, (3, 3), activation='relu')(input_layer)
    add = Concatenate()([avg_pool, conv])
    
    dense1 = Dense(256, activation='relu')(add)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
main_model = main_path(input_shape)
branch_model = branch_path(input_shape)
combined_model = keras.Model(inputs=[main_model.input, branch_model.input], outputs=[main_model.output, branch_model.output])

# Compile the model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return combined_model