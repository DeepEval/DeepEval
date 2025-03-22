import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D
from keras.models import Model

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: Second set of Convolutions
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Define a block with four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=2)(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = block(maxpool2)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)

    # Two sets of convolutions with skip connections
    def res_block(input_tensor, filters):
        conv = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        conv = BatchNormalization()(conv)
        conv = keras.activations.relu(conv)
        conv = Conv2D(filters, (3, 3), padding='same')(conv)
        conv = BatchNormalization()(conv)
        return keras.layers.add([conv, input_tensor])

    # Stage 3: Additional Convolutions and Skip Connections
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(block_output)
    conv4 = res_block(conv3, 128)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = res_block(maxpool3, 128)
    conv6 = res_block(conv5, 128)

    # Upsampling using UpSampling2D
    up_sample1 = UpSampling2D(size=(2, 2))(conv6)
    up_sample2 = UpSampling2D(size=(2, 2))(up_sample1)

    # Final Convolution and Output
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(up_sample2)
    output = Conv2D(units=10, kernel_size=(1, 1), activation='softmax')(conv7)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()