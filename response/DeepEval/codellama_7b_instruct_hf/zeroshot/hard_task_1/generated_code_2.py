from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation, Multiply
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_path1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_path1 = GlobalAveragePooling2D()(block1_path1)
    block1_path1 = Flatten()(block1_path1)
    block1_path1 = Dense(64, activation='relu')(block1_path1)
    block1_path1 = Dense(10, activation='softmax')(block1_path1)

    block1_path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_path2 = GlobalMaxPooling2D()(block1_path2)
    block1_path2 = Flatten()(block1_path2)
    block1_path2 = Dense(64, activation='relu')(block1_path2)
    block1_path2 = Dense(10, activation='softmax')(block1_path2)

    # Channel attention
    block1_channel_attention = Add()([block1_path1, block1_path2])
    block1_channel_attention = Activation('relu')(block1_channel_attention)
    block1_channel_attention = Dense(32, activation='relu')(block1_channel_attention)
    block1_channel_attention = Dense(32, activation='sigmoid')(block1_channel_attention)

    # Block 2
    block2_path1 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    block2_path1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(block2_path1)
    block2_path1 = Flatten()(block2_path1)
    block2_path1 = Dense(128, activation='relu')(block2_path1)
    block2_path1 = Dense(10, activation='softmax')(block2_path1)

    block2_path2 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    block2_path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(block2_path2)
    block2_path2 = Flatten()(block2_path2)
    block2_path2 = Dense(128, activation='relu')(block2_path2)
    block2_path2 = Dense(10, activation='softmax')(block2_path2)

    # Spatial attention
    block2_spatial_attention = Add()([block2_path1, block2_path2])
    block2_spatial_attention = Activation('relu')(block2_spatial_attention)
    block2_spatial_attention = Dense(64, activation='relu')(block2_spatial_attention)
    block2_spatial_attention = Dense(64, activation='sigmoid')(block2_spatial_attention)

    # Main path
    main_path = Add()([block1_path1, block1_path2])
    main_path = Activation('relu')(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Branch
    branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch = Flatten()(branch)
    branch = Dense(128, activation='relu')(branch)
    branch = Dense(10, activation='softmax')(branch)

    # Final classification
    final_classification = Add()([main_path, branch])
    final_classification = Activation('relu')(final_classification)
    final_classification = Dense(128, activation='relu')(final_classification)
    final_classification = Dense(10, activation='softmax')(final_classification)

    # Model
    model = Model(inputs=input_layer, outputs=final_classification)

    return model