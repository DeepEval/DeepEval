import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        # Average pooling with different scales
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        pool4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        # Flatten and concatenate
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat4 = Flatten()(pool4)
        concat = Concatenate()([flat1, flat2, flat4])

        return concat

    def block2(input_tensor):
        # Convolutional paths
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        # Average pooling with a 1x1 convolution
        avg_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        # Dropout for regularization
        path4_dropout = Dropout(0.5)(avg_pool)
        # Concatenate
        concat = Concatenate()([input_tensor, path4_dropout])

        return concat

    block1_output = block1(input_tensor=input_layer)
    concat_output = block2(input_tensor=block1_output)
    dense1 = Dense(units=128, activation='relu')(concat_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])