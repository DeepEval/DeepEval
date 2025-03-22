import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch Path
    branch_path = input_layer

    # Combine paths
    merged_path = Add()([conv2, branch_path]) 

    # Flatten and Dense Layer
    flatten_layer = Flatten()(merged_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model