import keras
from keras.layers import Input, Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # First sequential feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    batch_norm1 = BatchNormalization()(avg_pool1)

    # Second sequential feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(batch_norm1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    batch_norm2 = BatchNormalization()(avg_pool2)

    # Additional convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(batch_norm2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv3)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv4)

    # Another average pooling layer
    avg_pool3 = AveragePooling2D(pool_size=(2, 2))(conv5)
    batch_norm3 = BatchNormalization()(avg_pool3)

    # Flatten the feature maps
    flatten_layer = Flatten()(batch_norm3)

    # Two fully connected layers with dropout
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)

    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model