import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))

    # First feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv1)

    # Second feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2)

    # Third feature extraction layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv3)

    # Fourth feature extraction layer
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv4)

    # Flatten the feature maps
    flat = Flatten()(pool4)

    # Concatenate the feature maps from all layers
    concat = Concatenate()([flat, conv4, conv3, conv2, conv1])

    # First fully connected layer
    dense1 = Dense(units=1024, activation='relu')(concat)
    norm1 = BatchNormalization()(dense1)
    dropout1 = Dropout(rate=0.5)(norm1)

    # Second fully connected layer
    dense2 = Dense(units=512, activation='relu')(dropout1)
    norm2 = BatchNormalization()(dense2)
    dropout2 = Dropout(rate=0.5)(norm2)

    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()