import tensorflow as tf
from tensorflow.keras import layers, models
from .config import Config

def squeeze_excite_block(input_tensor, ratio=8):
    """ 
    Squeeze-and-Excitation Block.
    Explicitly models channel-wise dependencies.
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.Reshape((1, 1, channels))(x)
    
    # Excitation: Bottleneck -> Expansion with Sigmoid
    x = layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    # Scale original features
    return layers.multiply([input_tensor, x])

def residual_block(x, filters, downsample=False):
    """ 
    Residual block integrated with SE mechanism.
    """
    strides = 2 if downsample else 1
    shortcut = x
    
    # Conv Layer 1
    y = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    
    # Conv Layer 2
    y = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    
    # Attention Mechanism
    y = squeeze_excite_block(y)
    
    # Projection Shortcut if dimensions change
    if downsample:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    # Add residual connection
    y = layers.add([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_se_resnet(input_shape):
    """
    Constructs the full SE-ResNet Model.
    Structure: Entry Flow -> 3 SE-Residual Stages -> Regression Head
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- Entry Flow ---
    x = layers.Conv2D(Config.BASE_FILTERS, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # --- SE-Residual Stages ---
    # Stage 1 (Base Filters)
    x = residual_block(x, Config.BASE_FILTERS)
    x = residual_block(x, Config.BASE_FILTERS)
    
    # Stage 2 (2x Filters)
    x = residual_block(x, Config.BASE_FILTERS * 2, downsample=True)
    x = residual_block(x, Config.BASE_FILTERS * 2)
    
    # Stage 3 (4x Filters)
    x = residual_block(x, Config.BASE_FILTERS * 4, downsample=True)
    x = residual_block(x, Config.BASE_FILTERS * 4)
    
    # --- Regression Head ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Linear output for speed regression
    outputs = layers.Dense(1, activation='linear', dtype='float32')(x)
    
    return models.Model(inputs, outputs, name="SE_ResNet_Speed_Estimator")