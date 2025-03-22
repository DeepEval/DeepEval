import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten, Concatenate
from tensorflow.keras.applications import ResNet50

def dl_model():
    # Load a pre-trained ResNet50 model as a base model
    base_model = ResNet50(include_top=False, input_shape=(32, 32, 3))

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    weights = Dense(256, activation='relu')(x)
    weights = Dense(256, activation='sigmoid')(weights)
    weights = tf.reshape(weights, [-1, 32, 32, 3])
    main_path = tf.multiply(inputs, weights)

    # Branch path
    branch_path = Conv2D(3, kernel_size=3, padding='same')(inputs)

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined result
    x = Flatten()(combined)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()