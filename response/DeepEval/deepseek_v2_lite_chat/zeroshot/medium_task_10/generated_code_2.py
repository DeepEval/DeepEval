from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation, Concatenate, Add
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten

def dl_model(input_shape=(32, 32, 3)):
    # Input layer
    inputs = Input(shape=input_shape)

    # Base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze layers to prevent training during fine-tuning

    # Adjust the input feature dimensionality to 16
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Basic block
    main_path = Conv2D(64, (3, 3), activation='relu')(x)
    main_path = BatchNormalization()(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = BatchNormalization()(main_path)
    branch = Conv2D(64, (1, 1), activation='relu')(x)

    # Residual connection
    x = Add()([main_path, branch])
    x = Activation('relu')(x)

    # Second level residual blocks
    for _ in range(2):
        main_path = Conv2D(64, (3, 3), activation='relu')(x)
        main_path = BatchNormalization()(main_path)
        main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
        main_path = BatchNormalization()(main_path)
        branch = Conv2D(64, (1, 1), activation='relu')(x)
        x = Add()([main_path, branch])
        x = Activation('relu')(x)

    # Global branch for capturing initial convolution output
    global_branch = Conv2D(64, (1, 1), activation='relu')(inputs)
    global_branch = BatchNormalization()(global_branch)
    global_branch = Conv2D(64, (1, 1), activation='relu')(global_branch)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    return model