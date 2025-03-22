import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32 with 3 color channels

    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch Path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    
    # Reshape to match input for multiplication
    channel_weights = Dense(units=32 * 32 * 32, activation='sigmoid')(branch_path)
    channel_weights = keras.layers.Reshape((32, 32, 32))(channel_weights)

    # Multiply the channel weights with the input
    weighted_input = Multiply()([input_layer, channel_weights])

    # Combine paths
    combined_output = Add()([main_path, weighted_input])

    # Flatten and classify
    flatten_output = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model