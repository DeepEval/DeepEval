import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Define the model
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 3x32x32
    
    # Block 1: Global Pooling
    pool1 = GlobalAveragePooling2D()(input_layer)
    
    dense1 = Dense(units=128, activation='relu')(pool1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output1 = Dense(units=10, activation='softmax')(dense2)  # Output layer for Block 1
    
    # Block 2: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Connect Block 1 output to Block 2
    concat = keras.layers.concatenate([output1, pool2])
    
    # Additional fully connected layers
    dense3 = Dense(units=64, activation='relu')(concat)
    dense4 = Dense(units=32, activation='relu')(dense3)
    output2 = Dense(units=10, activation='softmax')(dense4)  # Output layer for Block 2
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=[output1, output2])
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1., 1.])