import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, LayerNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=1, kernel_size=(7, 7), strides=(1, 1), padding='same', depthwise_mode=True)(input_layer)
    main_path = LayerNormalization(epsilon=1e-6)(main_path)  # Layer normalization for standardization
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path (identity path)
    branch_path = input_layer

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model