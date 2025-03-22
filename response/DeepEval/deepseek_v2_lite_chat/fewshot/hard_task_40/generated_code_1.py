import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Average pooling layers
    def block_1(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        flat1 = Flatten()(avgpool3)
        flat2 = Flatten()(avgpool2)
        flat3 = Flatten()(avgpool1)
        concat = Concatenate()([flat1, flat2, flat3])
        return concat

    # Second block: Convolutional layers with different kernels
    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
        reshape = Reshape((-1, 16))(conv4)  # Reshape to 16-dimensional vector
        dense1 = Dense(units=128, activation='relu')(reshape)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        return output_layer

    # Connect the two blocks
    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=model)
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])