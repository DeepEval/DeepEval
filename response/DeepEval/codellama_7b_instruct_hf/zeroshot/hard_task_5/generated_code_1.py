from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, concatenate, Permute, Reshape, Add, Dense
from keras.applications import VGG16

def dl_model():
    # Block 1
    input_layer = Input(shape=(32, 32, 3))
    branch1 = input_layer
    branch1_split = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(branch1)
    branch1_conv = Conv2D(64, (1, 1), activation='relu')(branch1_split)
    branch1_conv_split = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(branch1_conv)
    branch1_fused = concatenate([branch1_conv_split], axis=3)
    
    # Block 2
    branch2 = branch1_fused
    branch2_reshaped = Reshape(target_shape=(32, 32, 3, 1))(branch2)
    branch2_permuted = Permute((3, 0, 1, 2))(branch2_reshaped)
    branch2_reshaped_back = Reshape(target_shape=(32, 32, 3))(branch2_permuted)
    branch2_shuffled = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(branch2_reshaped_back)
    branch2_conv = Conv2D(64, (1, 1), activation='relu')(branch2_shuffled)
    branch2_conv_split = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(branch2_conv)
    branch2_fused = concatenate([branch2_conv_split], axis=3)
    
    # Block 3
    branch3 = branch2_fused
    branch3_conv = Conv2D(64, (3, 3), activation='relu', use_bias=False, padding='same')(branch3)
    branch3_conv = Conv2D(64, (3, 3), activation='relu', use_bias=False, padding='same')(branch3_conv)
    branch3_conv = Conv2D(64, (3, 3), activation='relu', use_bias=False, padding='same')(branch3_conv)
    
    # Main path
    main_path = concatenate([branch1_fused, branch2_fused, branch3_conv], axis=3)
    
    # Branch
    branch = input_layer
    branch_conv = Conv2D(64, (1, 1), activation='relu')(branch)
    branch_pool = MaxPooling2D((2, 2))(branch_conv)
    
    # Combine
    combine = concatenate([main_path, branch_pool], axis=3)
    
    # Output
    output = Dense(10, activation='softmax')(combine)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    return model