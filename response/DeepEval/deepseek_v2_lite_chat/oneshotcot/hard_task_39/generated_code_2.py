import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), padding='same')(conv_output)
    pool1_flat = Flatten()(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_output)
    pool2_flat = Flatten()(pool2)
    
    pool4 = MaxPooling2D(pool_size=(4, 4), padding='same')(conv_output)
    pool4_flat = Flatten()(pool4)
    
    concat = Concatenate()([pool1_flat, pool2_flat, pool4_flat])
    
    reshape = Reshape((-1, 16))(concat)
    
    dense1 = Dense(units=128, activation='relu')(reshape)
    
    block2_input = dense1
    
    # Second block
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block2_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block2_input)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(block2_input)
    pool5 = MaxPooling2D(pool_size=(1, 1), padding='same')(conv3)
    
    pool5_flat = Flatten()(pool5)
    
    concat2 = Concatenate()([conv1, conv2, conv3, pool5_flat])
    
    dense2 = Dense(units=128, activation='relu')(concat2)
    dense3 = Dense(units=64, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model