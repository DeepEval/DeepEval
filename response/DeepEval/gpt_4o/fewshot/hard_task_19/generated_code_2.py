import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images
    
    # Main Path
    conv_main_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_1)
    conv_main_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_main_3)
    
    # Branch Path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense_branch_1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense_branch_2 = Dense(units=32*32*3, activation='sigmoid')(dense_branch_1)  # To generate weights matching the input shape
    channel_weights = Reshape(target_shape=(32, 32, 3))(dense_branch_2)
    weighted_input = Multiply()([input_layer, channel_weights])  # Apply the channel weights to the input
    
    # Adding Paths
    added_paths = Add()([main_path, weighted_input])
    
    # Fully Connected Layers for Classification
    flatten = Flatten()(added_paths)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model