import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_concat = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1_concat)
    conv2_concat = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2_concat)
    conv3_concat = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)

    # Pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv3_concat)

    # Flatten and fully connected layers
    flatten = Flatten()(pool)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()