import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 3: Add three max pooling layers
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Step 5: Flatten the outputs of each pooling layer
    flatten_1x1 = Flatten()(max_pooling_1x1)
    flatten_2x2 = Flatten()(max_pooling_2x2)
    flatten_4x4 = Flatten()(max_pooling_4x4)

    # Step 6: Concatenate the flattened outputs
    concatenated_features = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

    # Step 7: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(concatenated_features)
    
    # Step 8: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 9: Add output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model