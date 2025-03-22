import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Branch
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Branch Path
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
    adding_layer = Add()([main_path, branch_path])

    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(adding_layer)

    # Attention Mechanism
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    attention_weights = Dense(units=1, activation='softmax')(dense2)

    # Weighted Output
    weighted_output = Multiply()([adding_layer, attention_weights])
    flatten_layer = Flatten()(weighted_output)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model Definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model