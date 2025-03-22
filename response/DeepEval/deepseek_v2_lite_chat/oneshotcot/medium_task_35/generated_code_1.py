import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: Additional Convolution and MaxPooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Define a block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = block(pool2)
    batch_norm = BatchNormalization()(block_output)
    flatten = Flatten()(batch_norm)

    # Skip connections to restore spatial information
    skip_conn1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block_output)
    skip_conn2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch_norm)
    skip_conn3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(flatten)

    # Up-sampling and final convolution
    up_sample1 = UpSampling2D(size=(2, 2))(skip_conn1)
    merge1 = Concatenate()([up_sample1, skip_conn2, skip_conn3])
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(merge1)
    up_sample2 = UpSampling2D(size=(2, 2))(conv7)
    merge2 = Concatenate()([up_sample2, conv1])
    conv8 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid')(merge2)

    model = Model(inputs=input_layer, outputs=conv8)

    return model

# Build the model
model = dl_model()
model.summary()