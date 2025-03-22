import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial conv layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic Block Definition
    def basic_block(input_tensor):
      x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)
      x2 = input_tensor
      return Add()([x1, x2])

    # Main Path
    main_path = basic_block(x)
    main_path = basic_block(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Feature Fusion
    merged_path = Add()([main_path, branch_path])
    
    # Average Pooling
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(merged_path)

    # Flatten and FC Layers
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model