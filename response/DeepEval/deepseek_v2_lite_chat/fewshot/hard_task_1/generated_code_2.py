import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Concatenate, Activation, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32

    # Block 1: Feature Extraction
    conv1 = Conv2D(64, (3, 3), padding='same')(input_layer)  # Adjust the kernel size based on the number of input channels
    avg_pool = GlobalAveragePooling2D()(conv1)
    avg_pool_dense1 = Dense(128)(avg_pool)
    avg_pool_dense2 = Dense(10)(avg_pool)

    # Block 2: Feature Attention
    max_pool = GlobalMaxPooling2D()(conv1)
    max_pool_dense1 = Dense(128)(max_pool)
    max_pool_dense2 = Dense(10)(max_pool)

    # Concatenate the paths and add a 1x1 convolution for channel attention
    concat_layer = Concatenate()([avg_pool_dense1, max_pool_dense1])
    attn_dense1 = Dense(128)(concat_layer)
    attn_dense2 = Dense(1)(concat_layer)
    attn = Activation('sigmoid')(attn_dense2)  # Sigmoid activation for normalization

    # Element-wise multiplication between the original features and the attention weights
    attention_feature = avg_pool * attn
    attention_feature += max_pool * (1 - attn)

    # Spatial Feature Extraction
    avg_spatial_pool = AveragePooling2D(pool_size=(2, 2))(input_layer)
    max_spatial_pool = MaxPooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate the features from the previous blocks
    concat_spatial_feature = Concatenate()([avg_spatial_pool, max_spatial_pool])

    # Additional branch for channel matching
    additional_branch = Conv2D(1, (1, 1), padding='same')(input_layer)

    # Add the additional branch to the main feature path
    combined_feature = Add()([attention_feature, concat_spatial_feature, additional_branch])

    # Final classification
    classification_layer = Dense(10, activation='softmax')(combined_feature)

    # Model construction
    model = Model(inputs=input_layer, outputs=classification_layer)

    return model