import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Add second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Step 3: Add max-pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Step 4: Add the output features with the input layer
    added_features = Add()([max_pooling, input_layer])

    # Step 5: Add flatten layer
    flatten_layer = Flatten()(added_features)

    # Step 6: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 7: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 8: Add output layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can create an instance of the model like this:
model = dl_model()
model.summary()