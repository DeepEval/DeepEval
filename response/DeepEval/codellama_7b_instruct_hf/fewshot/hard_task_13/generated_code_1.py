import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    first_block = Input(shape=input_shape)
    branch_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(first_block)
    branch_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(first_block)
    branch_3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(first_block)
    branch_4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(first_block)
    concat = Concatenate()([branch_1, branch_2, branch_3, branch_4])
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)

    # Define the second block
    second_block = Input(shape=flatten.shape[1:])
    dense_1 = Dense(units=128, activation='relu')(second_block)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output = Dense(units=10, activation='softmax')(dense_2)

    # Define the model
    model = keras.Model(inputs=[first_block, second_block], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model