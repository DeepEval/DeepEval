import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Conv2DTranspose, Multiply, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    main_pool = MaxPooling2D(pool_size=(2, 2))(main_conv)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_pool)
    main_pool2 = MaxPooling2D(pool_size=(2, 2))(main_conv2)

    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch_pool = AveragePooling2D(pool_size=(3, 3))(branch_conv)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch_pool)
    branch_deconv1 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), activation='relu')(branch_conv2)
    branch_deconv2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(branch_deconv1)
    branch_deconv3 = Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(branch_deconv2)

    # Concatenate main and branch paths
    concatenated = Concatenate()([main_pool2, branch_deconv3])

    # Fully connected layer
    fc = Flatten()(concatenated)
    fc = Dense(units=128, activation='relu')(fc)
    output = Dense(units=10, activation='softmax')(fc)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])