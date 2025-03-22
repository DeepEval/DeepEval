import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, AveragePooling2D, Conv2DTranspose,
  Add, Activation, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv_in = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same",activation="relu")(input_layer)
    conv_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same",activation="relu")(conv_in)
    conv_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same",activation="relu")(conv_1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(conv_2)
    
    def block_channel_attention(input_tensor):
        shape_input = K.int_shape(input_tensor)
        shape_dense = tuple([shape_input[1], shape_input[2], shape_input[3] //16])
         
        avg_pool = AveragePooling2D(pool_size=(shape_input[1], shape_input[2]), strides=1, padding="valid")(input_tensor)
        conv_avg_pool = Conv2D(filters=shape_dense[2], kernel_size=(1, 1), strides=(1,1), padding="valid", activation="relu")(avg_pool)
        dense_avg_pool = Dense(units=shape_dense[2], activation="relu")(conv_avg_pool)
        dense_avg_pool_reshape = Dense(units=shape_dense[2])(dense_avg_pool)
        reshape_dense_avg_pool = Reshape(target_shape=shape_dense)(dense_avg_pool_reshape)
        upsample_dense_avg_pool = UpSampling2D(size=(shape_input[1] // shape_dense[1], shape_input[2] // shape_dense[2]))(reshape_dense_avg_pool)
        upsample_dense_avg_pool_channel = ExpandDims()(upsample_dense_avg_pool)
        upsample_dense_avg_pool_channel = Conv2D(filters=shape_input[3], kernel_size=(1,1), strides=(1, 1), padding="valid"(upsample_dense_avg_pool_channel)
        upsample_dense_avg_pool_channel = Conv2DTranspose(filters=shape_input[3],kernel_size=(1, 1), strides=(1, 1), padding="valid")(upsample_dense_avg_pool_channel)
        upsample_dense_avg_pool_channel = squeeze_excite(upsample_dense_avg_pool_channel)
        upsample_dense_avg_pool_channel = Reshape(target_shape=shape_input[1:])(upsample_dense_avg_pool_channel)
        scale_upsample_dense_avg_pool_channel = multiply([input_tensor, upsample_dense_avg_pool_channel])
         
        return scale_upsample_dense_avg_pool_channel
     
    avg_pool_channel = block_channel_attention(avg_pool)
     
    conv_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same",activation="relu")(avg_pool_channel)
    conv_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(conv_3)
     
    up_sample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(conv_4)
    up_sample = BatchNormalization()(up_sample)
    up_sample = Add()([up_sample, conv_2])
    up_sample = Activation("relu")(up_sample)
    up_sample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(up_sample)
    up_sample = BatchNormalization()(up_sample)
    up_sample = Add()([up_sample, conv_1])
    up_sample = Activation("relu")(up_sample)
    up_sample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2),padding="same", activation="relu")(up_sample)
    up_sample = BatchNormalization()(up_sample)
    up_sample = Add()([up_sample, conv_in])
    up_sample = Activation("relu")(up_sample)
    up_sample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same", activation="relu")(up_sample)
    up_sample = BatchNormalization()(up_sample)
    up_sample = Add()([up_sample, input_layer])
    up_sample = Activation("relu")(up_sample)
    up_sample = Conv2DTranspose(filters=10, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(up_sample)
    up_sample = BatchNormalization()(up_sample)
    up_sample = Flatten()(up_sample)
    up_sample = Dense(units=10, activation="softmax")(up_sample)
     
    model = keras.Model(inputs=input_layer, outputs=up_sample)

    return model