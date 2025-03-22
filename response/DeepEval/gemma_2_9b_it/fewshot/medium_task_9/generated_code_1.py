import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x

    # Main Path
    main_path = basic_block(conv1)
    main_path = basic_block(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Feature Fusion
    output_path = Add()([main_path, branch_path])

    # Final Branch
    branch_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_path)

    # Global Average Pooling
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8))(branch_output)

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model