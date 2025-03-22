import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    def block(input_tensor, dropout_rate):
        x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dropout(rate=dropout_rate)(x)
        return x

    def second_block(input_tensor):
        num_splits = 4
        splits = Lambda(lambda x: x[:, 0::4])(input_tensor)
        outputs = []
        for i in range(num_splits):
            filter_shape = (8, 8, 64, 64) if i == 0 else (32, 32, 64, 64)
            filter_shape += (1,) if i == 0 else (3, 3) if i == 1 else (5, 5) if i == 2 else (7, 7)
            conv = Conv2D(filters=filter_shape[3], kernel_size=filter_shape[1:], strides=filter_shape[3:])(splits[i])
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)
            outputs.append(conv)
        return Concatenate()(outputs)

    input_layer = Input(shape=(32, 32, 3))
    first_block_output = block(input_tensor=input_layer, dropout_rate=0.5)
    second_block_output = second_block(first_block_output)
    output_layer = Dense(units=10, activation='softmax')(second_block_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])