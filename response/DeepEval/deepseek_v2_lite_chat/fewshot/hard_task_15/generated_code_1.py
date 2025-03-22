import keras
from keras.models import Model
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Global average pooling
    main_pooling = AveragePooling2D(pool_size=(4, 4), padding='same')(input_layer)
    main_flatten = Flatten()(main_pooling)

    # Add fully connected layers
    main_dense1 = Dense(units=512, activation='relu')(main_flatten)
    main_dense2 = Dense(units=10, activation='softmax')(main_dense1)

    # Branch path: Direct connection
    branch_output = input_layer

    # Combine outputs of both paths
    combined_output = keras.layers.add([main_dense2, branch_output])

    # Add fully connected layers to the combined output
    combined_dense1 = Dense(units=512, activation='relu')(combined_output)
    combined_dense2 = Dense(units=10, activation='softmax')(combined_dense1)

    model = Model(inputs=input_layer, outputs=[main_dense2, combined_dense2])

    return model

# Instantiate the model
model = dl_model()
model.summary()