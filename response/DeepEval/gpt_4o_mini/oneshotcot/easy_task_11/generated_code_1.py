import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Step 1: Average Pooling Layer
    avg_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)

    # Step 2: 1x1 Convolutional Layer
    conv_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pooling)

    # Step 3: Flatten Layer
    flatten_layer = Flatten()(conv_layer)

    # Step 4: First Fully Connected Layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 5: Dropout Layer
    dropout_layer = Dropout(rate=0.5)(dense1)

    # Step 6: Second Fully Connected Layer
    dense2 = Dense(units=64, activation='relu')(dropout_layer)

    # Step 7: Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model