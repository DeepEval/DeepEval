import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import BatchNormalization, Activation

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

def dl_model():
    # Lambda layer to split the input into three groups
    input_tensor = Input(shape=input_shape)
    split1 = Lambda(lambda x: x[:, :, 0])(input_tensor)
    split2 = Lambda(lambda x: x[:, :, 1])(input_tensor)
    split3 = Lambda(lambda x: x[:, :, 2])(input_tensor)
    
    # Feature extraction layers
    conv1 = Conv2D(64, (1, 1), padding='same')(split1)
    conv2 = Conv2D(64, (3, 3), padding='same')(split2)
    conv3 = Conv2D(64, (5, 5), padding='same')(split3)
    
    # Max pooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the outputs
    fused = concatenate([pool1, pool2, pool3])
    
    # Flatten and fully connected layers
    flattened = Flatten()(fused)
    dense1 = Dense(128, activation='relu')(flattened)
    output = Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_tensor, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model