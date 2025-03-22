from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def channel_attention(input_tensor):
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool_flat = Flatten()(avg_pool)
    avg_pool_fc1 = Dense(64, activation='relu')(avg_pool_flat)
    avg_pool_fc2 = Dense(32, activation='relu')(avg_pool_fc1)
    avg_pool_fc3 = Dense(16, activation='softmax')(avg_pool_fc2)

    # Global max pooling
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool_flat = Flatten()(max_pool)
    max_pool_fc1 = Dense(64, activation='relu')(max_pool_flat)
    max_pool_fc2 = Dense(32, activation='relu')(max_pool_fc1)
    max_pool_fc3 = Dense(16, activation='softmax')(max_pool_fc2)

    # Element-wise multiplication
    attention_weights = Concatenate()([avg_pool_fc3, max_pool_fc3])
    attention_weights = BatchNormalization()(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = Dropout(0.5)(attention_weights)
    attention_output = input_tensor * attention_weights

    return attention_output

def spatial_attention(input_tensor):
    # Average pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2))(input_tensor)
    avg_pool_flat = Flatten()(avg_pool)
    avg_pool_fc1 = Dense(64, activation='relu')(avg_pool_flat)
    avg_pool_fc2 = Dense(32, activation='relu')(avg_pool_fc1)
    avg_pool_fc3 = Dense(16, activation='softmax')(avg_pool_fc2)

    # Max pooling
    max_pool = MaxPooling2D(pool_size=(2, 2))(input_tensor)
    max_pool_flat = Flatten()(max_pool)
    max_pool_fc1 = Dense(64, activation='relu')(max_pool_flat)
    max_pool_fc2 = Dense(32, activation='relu')(max_pool_fc1)
    max_pool_fc3 = Dense(16, activation='softmax')(max_pool_fc2)

    # Element-wise multiplication
    attention_weights = Concatenate()([avg_pool_fc3, max_pool_fc3])
    attention_weights = BatchNormalization()(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = Dropout(0.5)(attention_weights)
    attention_output = input_tensor * attention_weights

    return attention_output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Channel attention
    attention_output = channel_attention(input_layer)

    # Spatial attention
    attention_output = spatial_attention(attention_output)

    # Flatten and concatenate
    flatten_layer = Flatten()(attention_output)
    fc1 = Dense(128, activation='relu')(flatten_layer)
    fc2 = Dense(64, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model