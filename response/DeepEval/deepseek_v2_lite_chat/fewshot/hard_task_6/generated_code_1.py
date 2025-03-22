import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool1

    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool2

    def block3(input_tensor):
        conv1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool3

    def repeated_block1(input_tensor):
        return block1(input_tensor)

    branch_layer = Lambda(lambda x: MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)(name='Branch_Pooling')(input_layer))

    block1_output = repeated_block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)
    block3_output = block3(input_tensor=block2_output)
    branch_output = branch_layer

    concat_layer = Concatenate()([block3_output, branch_output])
    dense = Dense(units=256, activation='relu')(concat_layer)
    output = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Display the model summary
model.summary()