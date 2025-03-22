import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.initializers import he_normal

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    main_path_features = []

    for kernel_size in (1, 3, 5):
        conv = Conv2D(filters=32, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False, kernel_initializer=he_normal)(main_path)
        bn = BatchNormalization()(conv)
        main_path_features.append(bn)

    main_path_output = Concatenate(axis=3)(main_path_features)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=he_normal)(input_layer)
    branch_path_bn = BatchNormalization()(branch_path)

    # Output
    added_path = keras.layers.Add()([main_path_output, branch_path_bn])
    flatten_layer = Flatten()(added_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model