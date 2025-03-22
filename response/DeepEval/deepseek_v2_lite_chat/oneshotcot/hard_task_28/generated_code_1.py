import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Add, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming the input images are 32x32 pixels with 3 color channels
    
    # Main path
    main_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_norm = LayerNormalization()(main_conv)
    main_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_norm)
    main_conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_norm)
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine paths
    combined_tensor = Add()([main_conv2, branch_conv])
    combined_tensor = Concatenate(axis=-1)([combined_tensor, main_conv3])
    
    # Flatten and process through dense layers
    flatten = Flatten()(combined_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

model = dl_model()
model.summary()