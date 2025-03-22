import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, ZeroPadding2D, UpSampling2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # First branch: local feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    branch1 = AveragePooling2D(pool_size=(2, 2))(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)

    # Second branch: downsampling and upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=2)(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    # Third branch: downsampling and upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=2)(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate the outputs from each branch
    concat = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 convolutional layer for refinement
    conv_out = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)

    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(conv_out)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()