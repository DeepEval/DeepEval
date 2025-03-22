import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split input into three groups and process each group
    def block_1(input_tensor):
        split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_1[0])
        conv_3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_1[1])
        conv_5x5_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_1[2])
        batch_norm_1 = BatchNormalization()(conv_1x1_1)
        batch_norm_2 = BatchNormalization()(conv_3x3_1)
        batch_norm_3 = BatchNormalization()(conv_5x5_1)
        concat_1 = Concatenate(axis=-1)([batch_norm_1, batch_norm_2, batch_norm_3])
        return concat_1
    
    # Block 2: Four parallel branches
    def block_2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 1), padding='same')(path3)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        concat = Concatenate(axis=-1)([path1, path2, path3, path4])
        flatten = Flatten()(concat)
        dense = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(dense)
        return output
    
    # Connect the blocks
    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])