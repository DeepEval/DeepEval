import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Concatenate, Activation, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Channel Attention
    def channel_attention_module(input_tensor, ratio=16):
        avg_pool = AveragePooling2D(pool_size=(1, ratio))(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        concat = Concatenate()([avg_pool, max_pool])
        fc1 = Dense(units=input_tensor.shape[1] // 16, activation='relu', kernel_constraint=keras.constraints.MaxNorm(3))(concat)
        fc2 = Dense(units=input_tensor.shape[1], activation='sigmoid')(fc1)
        return Activation('relu')(input_tensor * fc2)

    # Block 2: Spatial Attention
    def spatial_attention_module(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(1, 7))(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        concat = Concatenate()([avg_pool, max_pool])
        fc = Dense(units=1, activation='sigmoid')(concat)
        return Activation('sigmoid')(input_tensor * fc)

    # Path 1: Global Average Pooling
    path1 = GlobalAveragePooling2D()(input_layer)
    path1_conv = Conv2D(filters=input_layer.shape[1], kernel_size=(1, 1), padding='same')(path1)
    path1_act = Activation('relu')(path1_conv)
    path1_fc1 = Dense(units=256, activation='relu')(path1_act)
    path1_fc2 = Dense(units=128, activation='relu')(path1_fc1)

    # Path 2: Global Max Pooling
    path2 = GlobalMaxPooling2D()(input_layer)
    path2_conv = Conv2D(filters=input_layer.shape[1], kernel_size=(1, 1), padding='same')(path2)
    path2_act = Activation('relu')(path2_conv)
    path2_fc1 = Dense(units=256, activation='relu')(path2_act)
    path2_fc2 = Dense(units=128, activation='relu')(path2_fc1)

    # Attention Layer
    attention_layer = Add()([path1_act, path2_act])
    attention_layer = Activation('sigmoid')(attention_layer)

    # Output Layer
    output = Dense(units=10, activation='softmax')(input_layer * attention_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model

model = dl_model()
model.summary()