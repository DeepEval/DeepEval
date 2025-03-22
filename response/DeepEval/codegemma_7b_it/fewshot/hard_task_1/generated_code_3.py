import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Activation, Conv2DTranspose, Reshape

def channel_attention(input_tensor, ratio):
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)
    avg_fc1 = Dense(int(input_tensor.shape[-1] // ratio), activation='relu')(avg_pool)
    avg_fc2 = Dense(int(input_tensor.shape[-1]), activation='sigmoid')(avg_fc1)
    max_fc1 = Dense(int(input_tensor.shape[-1] // ratio), activation='relu')(max_pool)
    max_fc2 = Dense(int(input_tensor.shape[-1]), activation='sigmoid')(max_fc1)
    multiply = Multiply()([avg_fc2, max_fc2])
    return multiply

def spatial_attention(input_tensor):
    avg_pool = AveragePooling2D()(input_tensor)
    max_pool = MaxPooling2D()(input_tensor)
    avg_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(avg_pool)
    max_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(max_pool)
    concat = Concatenate()([avg_conv, max_conv])
    upsample = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='sigmoid')(concat)
    upsample = Reshape((input_tensor.shape[1], input_tensor.shape[2], 1))(upsample)
    multiply = Multiply()([upsample, input_tensor])
    return multiply

def block_1(input_tensor):
    path1 = GlobalAveragePooling2D()(input_tensor)
    path1 = Dense(units=input_tensor.shape[-1] // 2, activation='relu')(path1)
    path1 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(path1)
    path2 = GlobalMaxPooling2D()(input_tensor)
    path2 = Dense(units=input_tensor.shape[-1] // 2, activation='relu')(path2)
    path2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(path2)
    attention_path = Add()([path1, path2])
    attention_path = Activation('sigmoid')(attention_path)
    attention_path = Reshape((input_tensor.shape[1], input_tensor.shape[2], 1))(attention_path)
    output = Multiply()([input_tensor, attention_path])
    return output

def block_2(input_tensor):
    avg_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    avg_path = AveragePooling2D()(avg_path)
    avg_path = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(avg_path)
    max_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    max_path = MaxPooling2D()(max_path)
    max_path = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(max_path)
    concat_path = Concatenate()([avg_path, max_path])
    output = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat_path)
    return output

def block_3(input_tensor):
    input_tensor_shape = input_tensor.shape
    residual_path = Conv2D(filters=input_tensor_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    channel_path = block_1(input_tensor)
    spatial_path = block_2(input_tensor)
    concat_path = Concatenate()([channel_path, spatial_path])
    output = Conv2D(filters=input_tensor_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_path)
    output = Add()([output, residual_path])
    output = Activation('relu')(output)
    return output

def dl_model():
    inputs = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    block_1_output = block_1(conv)
    block_2_output = block_2(block_1_output)
    block_3_output = block_3(block_2_output)
    flatten_layer = Flatten()(block_3_output)
    dense = Dense(units=10, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=inputs, outputs=dense)
    return model