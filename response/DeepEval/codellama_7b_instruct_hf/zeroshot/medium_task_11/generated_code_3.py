from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Concatenate
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    # Load the VGG16 model as a feature extractor
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the feature extractor layers
    for layer in base_model.layers:
        layer.trainable = False

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the convolutional layer
    conv_layer = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the parallel paths
    path1 = conv_layer
    path1 = GlobalAveragePooling2D()(path1)
    path1 = Dense(128, activation='relu')(path1)
    path1 = Dense(10, activation='softmax')(path1)

    path2 = conv_layer
    path2 = GlobalMaxPooling2D()(path2)
    path2 = Dense(128, activation='relu')(path2)
    path2 = Dense(10, activation='softmax')(path2)

    # Define the channel attention layer
    channel_attention = Add()([path1, path2])
    channel_attention = Activation('softmax')(channel_attention)

    # Define the spatial attention layer
    spatial_attention = Concatenate()([path1, path2])
    spatial_attention = GlobalAveragePooling2D()(spatial_attention)
    spatial_attention = Dense(128, activation='relu')(spatial_attention)
    spatial_attention = Dense(10, activation='softmax')(spatial_attention)

    # Define the fused feature map
    fused_features = Concatenate()([channel_attention, spatial_attention])
    fused_features = Flatten()(fused_features)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(fused_features)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model