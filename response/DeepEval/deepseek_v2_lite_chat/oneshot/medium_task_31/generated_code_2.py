import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input shape should match the dimensions of a single CIFAR-10 image
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the three paths for feature extraction
    def conv_path(split):
        path = []
        for i in range(3):
            conv = Conv2D(filters=64, kernel_size=(kernel_size[i], kernel_size[i]), strides=(1, 1), padding='same', activation='relu')(split[i])
            path.append(conv)
        return path
    
    # Convolution paths
    paths = conv_path(split_1)
    
    # Concatenate the outputs of the three paths
    concatenated = Concatenate()(paths)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()