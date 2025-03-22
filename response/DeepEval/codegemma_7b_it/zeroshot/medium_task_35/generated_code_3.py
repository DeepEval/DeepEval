from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Input, concatenate

def dl_model():
    # Define the input layer
    img_input = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv_1 = Conv2D(64, (3, 3), activation='relu')(img_input)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Stage 2: Feature Extraction
    conv_3 = Conv2D(256, (3, 3), activation='relu')(pool_2)
    conv_4 = Conv2D(256, (3, 3), activation='relu')(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu')(pool_3)
    conv_6 = Conv2D(512, (3, 3), activation='relu')(conv_5)
    drop_6 = Dropout(0.5)(conv_6)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(drop_6)

    # Stage 3: Upsampling and Fusion
    up_7 = UpSampling2D(size=(2, 2))(pool_4)
    merge_7 = concatenate([drop_6, up_7], axis=3)
    conv_7 = Conv2D(512, (3, 3), activation='relu')(merge_7)

    up_8 = UpSampling2D(size=(2, 2))(conv_7)
    merge_8 = concatenate([conv_6, up_8], axis=3)
    conv_8 = Conv2D(256, (3, 3), activation='relu')(merge_8)

    up_9 = UpSampling2D(size=(2, 2))(conv_8)
    merge_9 = concatenate([conv_4, up_9], axis=3)
    conv_9 = Conv2D(256, (3, 3), activation='relu')(merge_9)

    up_10 = UpSampling2D(size=(2, 2))(conv_9)
    merge_10 = concatenate([conv_2, up_10], axis=3)
    conv_10 = Conv2D(128, (3, 3), activation='relu')(merge_10)

    up_11 = UpSampling2D(size=(2, 2))(conv_10)
    merge_11 = concatenate([conv_1, up_11], axis=3)
    conv_11 = Conv2D(64, (3, 3), activation='relu')(merge_11)

    # Output layer
    conv_12 = Conv2D(10, (1, 1), activation='sigmoid')(conv_11)

    # Create the model
    model = Model(img_input, conv_12)

    return model