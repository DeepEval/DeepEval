import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the main path
def main_path(input_shape):
    x = Input(shape=input_shape)
    split_1 = Lambda(lambda x: x[:, :16]])(x)
    split_2 = Lambda(lambda x: x[:, 16:32])(x)
    split_3 = Lambda(lambda x: x[:, 32:48])(x)
    
    conv1_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_1)
    conv1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_2)
    conv1_3 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_3)
    
    # Concatenate the outputs from the main path
    concat = Concatenate()([conv1_1, conv1_2, conv1_3])
    
    return Model(inputs=[x], outputs=[concat])

# Define the branch path
def branch_path(input_shape):
    x = Input(shape=input_shape)
    conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    
    # Align the number of output channels with those of the main path
    conv1.shape[3]
    
    # Pass the output of the branch path through a fully connected layer for classification
    output = Flatten()(conv1)
    output = Dense(1024, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)
    
    return Model(inputs=[x], outputs=[output])

# Define the fused model
def dl_model():
    input_shape = (32, 32, 3)  # Assuming the input images are 32x32 pixels
    
    # Main path
    main_path_output = main_path(input_shape)(x_train)
    main_model = Model(inputs=x_train, outputs=main_path_output)
    
    # Branch path
    branch_path_output = branch_path(input_shape)(main_path_output)
    branch_model = Model(inputs=x_train, outputs=branch_path_output)
    
    # Combine the outputs of the main path and the branch path
    fused_model = Model(inputs=main_path_input, outputs=Concatenate()([main_path_output, branch_path_output]))
    
    # Compile the model
    fused_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return fused_model

# Create the model
model = dl_model()

# Print the model summary
model.summary()