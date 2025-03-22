import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Max pooling
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        # Convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
        # Max pooling
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        # Convolutional layer
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
        
        # Concatenate features from the different layers
        concat = Concatenate()(inputs=[conv3, conv2, conv1, input_tensor])
        return concat
    
    # Block 2
    def block2(input_tensor):
        # Convolutional layer
        conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Convolutional layer
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Convolutional layer
        conv6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Max pooling
        maxpool3 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        # Concatenate features from the different layers
        concat = Concatenate()(inputs=[conv6, conv5, conv4, maxpool3])
        return concat
    
    # Apply blocks
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    
    # Flatten and feed into dense layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.summary()