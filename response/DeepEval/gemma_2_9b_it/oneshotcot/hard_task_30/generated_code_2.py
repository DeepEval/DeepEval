import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Dual-path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    block1_output = Concatenate()([main_path, branch_path])

    # Block 2: Split-and-Concatenate
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(block1_output)
    
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    block2_output = Concatenate()([group1, group2, group3])

    # Classifier
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model