import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Number of classes
num_classes = y_train.shape[1]

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_input)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    main_output = MaxPooling2D()(x)
    
    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(64, kernel_size=(5, 5), activation='relu')(branch_input)
    branch_output = MaxPooling2D()(branch_x)
    
    # Concatenate the outputs of the main and branch paths
    concatenated = Concatenate()([main_output, branch_output])
    
    # Flatten and add fully connected layers
    flattened = Flatten()(concatenated)
    output = Dense(num_classes, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Return the constructed model
return dl_model()