import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction paths
    fe_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_split[0])
    fe_path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_split[1])
    fe_path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_split[2])
    
    # Concatenate outputs from main paths
    concat = Concatenate()(fe_path1)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = Flatten()(branch_conv)
    
    # Combine main path and branch path outputs
    fused_features = Concatenate()([concat, branch_output])
    
    # Classification layer
    dense1 = Dense(units=128, activation='relu')(fused_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()