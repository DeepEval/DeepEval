import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Flatten
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # Input shape for CIFAR-10
    num_classes = 10  # Number of output classes for CIFAR-10

    # Stage 1: Downsampling using Conv2D and MaxPooling2D
    downsampling_inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(downsampling_inputs)
    x = MaxPooling2D()(x)
    x = MaxPooling2D()(x)
    x = MaxPooling2D()(x)

    # Stage 2: Encoder block for feature extraction
    encoder_block = Conv2D(128, (3, 3), activation='relu')(x)
    encoder_block = MaxPooling2D()(encoder_block)

    # Stage 3: Decoder block for upsampling
    decoder_block = Conv2DTranspose(64, (3, 3), strides=2, activation='relu')(encoder_block)
    decoder_block = Conv2D(64, (3, 3), activation='relu')(decoder_block)
    decoder_block = UpSampling2D()(decoder_block)

    # Skip connection to restore spatial information
    skip_connection = encoder_block
    decoder_block = Concatenate()([decoder_block, skip_connection])

    # Additional convolution and upsampling for better restoration
    decoder_block = Conv2D(64, (3, 3), activation='relu')(decoder_block)
    decoder_block = UpSampling2D()(decoder_block)
    decoder_block = Conv2D(3, (3, 3), activation='sigmoid')(decoder_block)

    # Output layer for classification
    output = Conv2D(num_classes, (1, 1), activation='softmax')(decoder_block)

    # Model construction
    model = Model(inputs=downsampling_inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()
print(model.summary())