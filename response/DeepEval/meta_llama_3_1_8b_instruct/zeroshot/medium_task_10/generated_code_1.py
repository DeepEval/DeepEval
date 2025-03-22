from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3), name='input_layer')

    # First level residual connection
    x = Conv2D(16, (3, 3), padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)

    # Basic block
    def basic_block(x):
        branch = Conv2D(16, (3, 3), padding='same', name='conv_branch')(x)
        branch = BatchNormalization(name='bn_branch')(branch)
        branch = ReLU(name='relu_branch')(branch)

        main_path = Conv2D(16, (3, 3), padding='same', name='conv_main')(x)
        main_path = BatchNormalization(name='bn_main')(main_path)
        main_path = ReLU(name='relu_main')(main_path)

        return Add(name='add_block')([main_path, branch])

    # Second level residual connection
    x = basic_block(x)
    x = basic_block(x)

    # Third level residual connection
    x = Conv2D(32, (3, 3), padding='same', name='global_conv')(x)
    x = BatchNormalization(name='global_bn')(x)
    x = ReLU(name='global_relu')(x)

    # Add the global branch to the second-level residual structure
    x = Add(name='final_add')([x, x])

    # Average pooling and fully connected layer
    x = AveragePooling2D(name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='image_classification_model')

    return model

# Usage
model = dl_model()
model.summary()