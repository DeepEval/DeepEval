import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Two <convolution, convolution, max pooling> blocks
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        return conv2, pool2

    # Block 2: Two <convolution, convolution, convolution, max pooling> blocks
    def block2(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        conv4 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        return conv4, pool4

    # Merge the feature maps from both blocks
    conv2, pool2 = block1(input_layer)
    conv4, pool4 = block2(conv2)
    concat = Concatenate()([conv4, pool4])

    # Flatten the merged feature maps
    flatten = Flatten()(concat)

    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])