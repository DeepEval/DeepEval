import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv2)
    main_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(main_conv3)

    # Branch path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    branch_dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    branch_dense2 = Dense(units=32, activation='relu')(branch_dense1)

    # Generate channel weights
    channel_weights = Dense(units=3, activation='sigmoid')(branch_dense2)
    reshaped_weights = Reshape((1, 1, 3))(channel_weights)
    scaled_input = Multiply()([input_layer, reshaped_weights])

    # Merge main path and branch path
    added_outputs = Add()([main_pooling, scaled_input])

    # Final fully connected layers for classification
    flatten_layer = Flatten()(added_outputs)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model