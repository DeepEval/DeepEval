import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Reshape, Permute, Add, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split input into three groups and apply 1x1 convolutions
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv_blocks = [Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in split_layer]
    fused_features = Concatenate()(conv_blocks)

    # Block 2: Channel shuffle and reshape
    reshape_layer = Lambda(lambda tensors: tf.transpose(tensors, perm=[0, 3, 1, 2]))(fused_features)
    reshape_layer = Lambda(lambda tensors: tf.reshape(tensors, (tf.shape(tensors)[0], tf.shape(tensors)[1], tf.shape(tensors)[3]*tf.shape(tensors)[2])))
    shuffled_features = reshape_layer

    # Block 3: Depthwise separable convolution
    dw_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_depthwise=True)(shuffled_features)
    dw_conv = BatchNormalization()(dw_conv)
    dw_conv = Activation('relu')(dw_conv)
    
    # Branch: Direct input without transformation
    direct_input = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs from main path and branch
    combined = Add()([dw_conv, direct_input])
    
    # Final dense layers for classification
    dense = Dense(units=10, activation='softmax')(combined)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()
model.summary()