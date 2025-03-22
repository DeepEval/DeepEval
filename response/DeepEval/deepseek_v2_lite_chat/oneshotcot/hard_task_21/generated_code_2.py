import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups for the main path
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(group1)
    conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(group2)
    conv3 = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(group3)
    
    # Concatenate the outputs of the main path
    main_output = Concatenate()([conv1, conv2, conv3])
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(main_output)
    
    # Add the outputs of the main and branch paths
    combined_output = keras.backend.sum([main_output, branch_conv], axis=0)
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(combined_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model