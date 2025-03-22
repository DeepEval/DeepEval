import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(224, 224, 3))

    # Step 2: First feature extraction layer (Convolution + Average Pooling)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    # Step 3: Second feature extraction layer (Convolution + Average Pooling)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Step 4: Additional Convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    # Step 5: Final Average Pooling layer to reduce dimensionality
    avg_pool3 = AveragePooling2D(pool_size=(2, 2))(conv5)

    # Step 6: Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)

    # Step 7: First fully connected layer with dropout
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.5)(dense1)

    # Step 8: Second fully connected layer with dropout
    dense2 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)

    # Step 9: Output layer for classification with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model