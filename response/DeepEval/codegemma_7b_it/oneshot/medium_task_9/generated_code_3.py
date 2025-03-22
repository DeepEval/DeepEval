import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block 1
    conv1_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    batch_norm1 = BatchNormalization()(conv1_path)
    relu1 = keras.activations.relu(batch_norm1)

    conv2_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu1)
    batch_norm2 = BatchNormalization()(conv2_path)
    relu2 = keras.activations.relu(batch_norm2)

    branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    feature_fusion = Add()([relu2, branch_path])

    # Basic block 2
    conv3_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(feature_fusion)
    batch_norm3 = BatchNormalization()(conv3_path)
    relu3 = keras.activations.relu(batch_norm3)

    conv4_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu3)
    batch_norm4 = BatchNormalization()(conv4_path)
    relu4 = keras.activations.relu(batch_norm4)

    branch_path2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(feature_fusion)
    feature_fusion2 = Add()([relu4, branch_path2])

    # Pooling and flattening
    avg_pool = MaxPooling2D(pool_size=(2, 2))(feature_fusion2)
    flatten_layer = Flatten()(avg_pool)

    # Fully connected layer
    dense = Dense(units=64, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model