import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First part: Feature extraction with convolutional and max-pooling layers
    conv_pool1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_pool1_1)
    
    conv_pool2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_pool1_2)
    conv_pool2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_pool2_1)
    
    conv_pool3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_pool2_2)
    conv_pool3_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_pool3_1)
    
    # Second part: Enhance generalization with convolutional, dropout, and convolutional layers
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_pool3_2)
    dropout = Dropout(0.5)(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)
    
    # Third part: Reconstruction with convolutional and transposed convolutional layers, using skip connections
    upsample1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)
    concat1 = Add()([upsample1, conv_pool2_1])  # Skip connection from conv_pool2_1
    conv_trans1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv_trans1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_trans1_1)
    
    upsample2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_trans1_2)
    concat2 = Add()([upsample2, conv_pool1_1])  # Skip connection from conv_pool1_1
    conv_trans2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    conv_trans2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_trans2_1)
    
    upsample3 = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_trans2_2)
    concat3 = Add()([upsample3, input_layer])  # Skip connection from input_layer
    final_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat3)
    
    # Output part: Generate probability output for 10 classes
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(final_conv)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model