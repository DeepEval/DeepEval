import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, AveragePooling2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Multiply, Add, Lambda
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=input_layer.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    def block1(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=64, activation='relu')(path1)
        path1 = Dense(units=10, activation='softmax')(path1)
        
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=64, activation='relu')(path2)
        path2 = Dense(units=10, activation='softmax')(path2)
        
        return Add()([path1, path2])
    
    block1_output = block1(conv)
    channel_attention_weights = Lambda(lambda x: keras.backend.repeat_elements(keras.backend.expand_dims(x, axis=-1), x.shape[-1], axis=-1))(block1_output)
    channel_attention_weights = Multiply()([channel_attention_weights, conv])
    block1_output = Multiply()([channel_attention_weights, conv])

    def block2(input_tensor):
        path1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        output_tensor = Concatenate()([path1, path2])
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(output_tensor)
        return output_tensor
        
    block2_output = block2(block1_output)
    channel_attention_weights = Multiply()([block2_output, block1_output])
    final_features = Add()([channel_attention_weights, block1_output])

    final_conv = Conv2D(filters=final_features.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(final_features)
    output_layer = Flatten()(final_conv)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])