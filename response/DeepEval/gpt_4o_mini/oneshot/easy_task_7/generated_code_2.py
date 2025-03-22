import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.2)(main_path)
    
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(0.2)(main_path)
    
    main_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined = Add()([main_path, branch_path])

    # Flatten and Dense layers
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model