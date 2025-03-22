import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block:
    
    group_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(group_input[0])
    conv2 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(group_input[1])
    conv3 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x))(group_input[2])
    
    concat_layer1 = Concatenate()([conv1, conv2, conv3])
    
    # Second Block:
    
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_layer1)
    
    branch_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(conv4)
    
    branch_conv1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(branch_input[0])
    branch_conv2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(branch_input[1])
    branch_conv3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(branch_input[2])
    
    branch_conv4 = Lambda(lambda x: MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x))(branch_input[0])
    
    concat_layer2 = Concatenate()([branch_conv1, branch_conv2, branch_conv3, branch_conv4])
    
    # Final Layer:
    
    global_avg_pool = GlobalAveragePooling2D()(concat_layer2)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model