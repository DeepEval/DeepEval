import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Pathway 1
    path1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv1)
    path1_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1_conv2)

    # Pathway 2
    path2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine pathways
    combined = Add()([path1_avg_pool, path2_conv])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layer to map to class probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model