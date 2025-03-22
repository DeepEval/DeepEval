import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.merge import Add
from keras.layers import GlobalAveragePooling2D

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the architecture of the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional branch
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second convolutional branch
    conv2 = Conv2D(64, kernel_size=(5, 5), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Add the outputs of the two branches
    add_layer = Add()([maxpool1, maxpool2])
    
    # Global average pooling to compress features
    avgpool = GlobalAveragePooling2D()(add_layer)
    
    # Fully connected layer
    fc1 = Dense(128, activation='relu')(avgpool)
    fc2 = Dense(10, activation='softmax')(fc1)  # 10 classes for CIFAR-10
    
    # Two separate paths for the outputs of the two branches
    branch1_output = Dense(1, name='branch1_output')(fc1)
    branch2_output = Dense(1, name='branch2_output')(fc2)
    
    # Generate attention weights based on the outputs of the two branches
    attention_weight = keras.layers.Lambda(lambda x: x[0])(
        keras.layers.Activation('sigmoid'))(
        keras.layers.concatenate([branch1_output, branch2_output]))
    
    # Weighted addition of the two branches
    output = keras.layers.Lambda(lambda x, w: x * w)(
        inputs=[add_layer, attention_weight[:, 0] * branch1_output + attention_weight[:, 1] * branch2_output])
    
    # Return the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()