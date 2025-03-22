import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    main_path = Concatenate()([conv1_1, conv1_2, conv1_3])

    # Branch path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = conv2_1

    # Combine paths
    fused_features = Add()([main_path, branch_path])

    # Classification layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model