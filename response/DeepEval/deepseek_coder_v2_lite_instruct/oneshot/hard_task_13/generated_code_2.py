import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = first_block(input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(batch_norm1)

    # Second block
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Final reshaping and multiplication
    reshape_layer = Reshape((32, 32, 3))(output_layer)
    elementwise_product = keras.layers.multiply([input_layer, reshape_layer])
    final_flatten = Flatten()(elementwise_product)
    final_dense = Dense(units=10, activation='softmax')(final_flatten)

    model = Model(inputs=input_layer, outputs=final_dense)

    return model

# Create the model
model = dl_model()
model.summary()