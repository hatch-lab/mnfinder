import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.keras import Model

# Adapted from 
# https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py
class AttentionUNet:
  """
  Builds a standard U-Net with attention in the up-blocks
  """
  def build(self, crop_size, num_input_channels, num_output_classes, depth=4):
    """
    Build and return the network model

    Input assumes channels are last, with image shapes being
    (crop_size, crop_size, num_input_channels)

    Parameters
    --------
    crop_size : int
      The width and height of each image
    num_input_channels : int
      The number of channels
    num_output_classes : int
      The number of output classes
    depth : int
      The depth of the neural net
    
    Returns
    --------
    tf.keras.Model
      The built model
    """
    inputs = nn.Input(( crop_size, crop_size, num_input_channels ))
    x = inputs

    features = 64
    skips = []
    for i in range(depth):
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      x = nn.Dropout(0.2)(x)
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      skips.append(x)
      x = nn.MaxPooling2D((2, 2), data_format='channels_last')(x)
      features *= 2

    # Bottleneck
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)
    x = nn.Dropout(0.2)(x)
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)

    for i in reversed(range(depth)):
      features = features // 2
      x = self._attention_up_and_concat(x, skips[i], data_format='channels_last')
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      x = nn.Dropout(0.2)(x)
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    conv6 = nn.Conv2D(num_output_classes, (1, 1), padding='same', data_format='channels_last')(x)
    conv7 = nn.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

  def _attention_up_and_concat(self, down_layer, layer, data_format='channels_last'):
    """
    Build the attention+up block

    Parameters
    --------
    down_layer : tf.layers.Layer
      The layer to up-sample
    layer : tf.layers.Layer
      The down-layer to make a skip connection
    data_format : str
      Can be 'channels_last' or 'channels_first'
    
    Returns
    --------
    tf.layers.Layer
      The concatenated up-sampled and skip-connection layers
    """
    if data_format == 'channels_first':
      in_channel = down_layer.get_shape().as_list()[1]
    else:
      in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = nn.UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = self._attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    concat = nn.Concatenate()([up, layer])
    return concat

  def _attention_block_2d(self, x, g, inter_channel, data_format='channels_last'):
    """
    Build the attention block

    From 
    https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py

    Parameters
    --------
    x : tf.layers.Layer
      The skip connection
    g : tf.layers.Layer
      The up-sampled layer
    inter_channel : int
      The number of channels
    data_format : str
      Can be 'channels_last' or 'channels_first'
    
    Returns
    --------
    tf.layers.Layer
      The attention block
    """
    # theta_x(?,g_height,g_width,inter_channel)
    theta_x = nn.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = nn.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)
    f = nn.Activation('relu')(nn.Add()([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)
    psi_f = nn.Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = nn.Activation('sigmoid')(psi_f)
    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)
    att_x = nn.Multiply()([x, rate])

    return att_x

class MSAttentionUNet(AttentionUNet):
  """
  Builds a modified U-Net with attention in the up-blocks

  Each downblock is modified to perform a multi-scale convolution:
  inputs are sent in parallel through 3 convolution layers:
    * strides (1,1)
    * strides (2,2) -> upsampled(2,2)
    * strides (4,4) -> upsampled(4,4)
  The 3 layers are concatenated and the maxi intensity projection is taken

  """
  def build(self, crop_size, num_input_channels, num_output_classes, depth=4):
    """
    Build and return the network model

    Input assumes channels are last, with image shapes being
    (crop_size, crop_size, num_input_channels)

    Parameters
    --------
    crop_size : int
      The width and height of each image
    num_input_channels : int
      The number of channels
    num_output_classes : int
      The number of output classes
    depth : int
      The depth of the neural net
    
    Returns
    --------
    tf.keras.Model
      The built model
    """
    inputs = nn.Input(( crop_size, crop_size, num_input_channels ))
    x = inputs

    features = 64
    skips = []
    for i in range(depth-1):
      x = self.ms_down_block(x, features, 'channels_last')
      skips.append(x)
      features *= 2
    x = self.ms_down_block(x, features, 'channels_last')
    features *= 2

    # Bottleneck
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)
    x = nn.Dropout(0.2)(x)
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)

    for i in reversed(range(1,depth)):
      features = features // 2
      x = self._attention_up_and_concat(x, skips[i-1], data_format='channels_last')
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      x = nn.Dropout(0.2)(x)
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    x = nn.Conv2DTranspose(features, 3, 2, padding='same')(x)
    x = nn.Dropout(0.2)(x)
    x = nn.Conv2D(features, (3,3), padding='same', activation='relu',)(x)
    x = nn.Conv2D(features, (3,3), padding='same', activation='relu',)(x)

    conv6 = nn.Conv2D(num_output_classes, (1, 1), padding='same', data_format='channels_last')(x)
    conv7 = nn.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

  def ms_down_block(self, x, features, data_format='channels_last'):
    """
    Build the multi-scale down block

    Parameters
    --------
    x : tf.layers.Layer
      The input layer
    features : int
      The number of features
    data_format : str
      Can be 'channels_last' or 'channels_first'
    
    Returns
    --------
    tf.layers.Layer
      The concatenated up-sampled and skip-connection layers
    """
    x1 = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x2 = nn.Conv2D(features, (3, 3), strides=(2,2), activation='relu', padding='same', data_format=data_format)(x)
    x3 = nn.Conv2D(features, (3, 3), strides=(4,4), activation='relu', padding='same', data_format=data_format)(x)

    x2 = nn.UpSampling2D(size=(2,2), data_format=data_format, interpolation='bilinear')(x2)
    x3 = nn.UpSampling2D(size=(4,4), data_format=data_format, interpolation='bilinear')(x3)

    x = nn.Concatenate()([x1, x2, x3])
    x = nn.Dropout(0.2)(x)
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)
    
    x = nn.Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
    x = nn.MaxPooling3D((2,2,3), data_format=data_format)(x)
    x = nn.Lambda(lambda x: tf.squeeze(x, axis=-1))(x)
    return x

class SegmenterUNet(AttentionUNet):
  """
  Builds a standard U-Net with attention in the up-blocks
  """
  def build(self, crop_size, num_input_channels, depth=4):
    """
    Build and return the network model

    Input assumes channels are last, with image shapes being
    (crop_size, crop_size, num_input_channels)

    Parameters
    --------
    crop_size : int
      The width and height of each image
    num_input_channels : int
      The number of channels
    num_output_classes : int
      The number of output classes
    depth : int
      The depth of the neural net
    
    Returns
    --------
    tf.keras.Model
      The built model
    """
    inputs = nn.Input(( crop_size, crop_size, num_input_channels ))
    x = inputs

    features = 64
    skips = []
    for i in range(depth):
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      x = nn.Dropout(0.2)(x)
      x = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
      skips.append(x)
      x = nn.MaxPooling2D((2, 2), data_format='channels_last')(x)
      features *= 2

    # Bottleneck
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)
    x = nn.Dropout(0.2)(x)
    x = nn.Conv2D(features, (3,3), activation='relu', padding='same', data_format='channels_last')(x)

    hull_features = features
    hull_decoder = x
    hull_skips = []
    for i in reversed(range(depth)):
      hull_features = hull_features // 2
      hull_decoder = self._attention_up_and_concat(hull_decoder, skips[i], data_format='channels_last')
      hull_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(hull_decoder)
      hull_decoder = nn.Dropout(0.2)(hull_decoder)
      hull_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(hull_decoder)
      hull_skips.insert(0, hull_decoder)
    hull_decoder = nn.Dense(units=1)(hull_decoder)

    # seg_features = features
    # seg_decoder = x
    # seg_skips = []
    # for i in reversed(range(depth)):
    #   seg_features = seg_features // 2
    #   seg_decoder = self._attention_up_and_concat(seg_decoder, skips[i], data_format='channels_last')
    #   # seg_decoder = nn.Add()([seg_decoder, hull_skips[i]])
    #   seg_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(seg_decoder)
    #   seg_decoder = nn.Dropout(0.2)(seg_decoder)
    #   seg_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(seg_decoder)

    # seg_decoder = nn.Conv2D(num_output_classes, (1, 1), padding='same', data_format='channels_last')(seg_decoder)
    # seg_decoder = nn.Activation('sigmoid')(seg_decoder)

    # concat_block = nn.Concatenate()([ hull_decoder, seg_decoder ])
    model = Model(inputs=inputs, outputs=hull_decoder)

    return model

class UNet3(AttentionUNet):
  """
  Builds a standard U-Net with attention in the up-blocks
  """
  def build(self, crop_size, num_input_channels, num_output_classes=1, depth=4, training=False):
    """
    Build and return the network model

    Input assumes channels are last, with image shapes being
    (crop_size, crop_size, num_input_channels)

    Parameters
    --------
    crop_size : int
      The width and height of each image
    num_input_channels : int
      The number of channels
    num_output_classes : int
      The number of output classes
    depth : int
      The depth of the neural net
    
    Returns
    --------
    tf.keras.Model
      The built model
    """
    inputs = nn.Input(( crop_size, crop_size, num_input_channels ))

    # Keep number of features constant to reduce training params
    # per half u-net paper
    # 0       0 128
    #  1     1  64
    #   2   2   32
    #    3 3    16
    #     4     8

    features = 128

    # Encoder
    encoder_blocks = []
    e1 = self._make_conv_block(inputs, features) # 128x128x64
    encoder_blocks.append(e1)

    for i in range(1,depth):
      encoder_blocks.append(nn.MaxPooling2D((2,2))(encoder_blocks[i-1]))
      encoder_blocks[-1] = self._make_conv_block(encoder_blocks[-1], features)
    
    # Decoder
    # Dual U-Net design
    # hull_decoder = encoder_blocks[-1]
    # # hull_features = features
    # for i in reversed(range(depth-1)):
    #   # hull_features = hull_features // 2
    #   hull_decoder = self._attention_up_and_concat(hull_decoder, encoder_blocks[i], data_format='channels_last')
    #   hull_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(hull_decoder)
    #   hull_decoder = nn.Dropout(0.2)(hull_decoder)
    #   hull_decoder = nn.Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_last')(hull_decoder)
    # hull_decoder = nn.Dense(units=1)(hull_decoder)

    decoder_blocks = []
    for i in reversed(range(depth-1)):
      # features = features // 2
      # Skip connections
      down_skips = []
      down_skips.append(self._make_conv_block(encoder_blocks[i], features))
      for j in range(1,i+1):
        down_skips.append(nn.MaxPool2D((2**j, 2**j))(encoder_blocks[i-j]))
        down_skips[-1] = self._make_conv_block(down_skips[-1], features, n=1)

      decoder_skips = []
      for j in range(1,len(decoder_blocks)):
        decoder_skips.append(nn.UpSampling2D((2**(j+1),2**(j+1)))(decoder_blocks[j]))

      up_skips = []
      for j in range(i+1,depth-1):
        up_skips.append(nn.UpSampling2D((2**(j-i), 2**(j-i)))(encoder_blocks[j]))
        up_skips[-1] = self._make_conv_block(up_skips[-1], features, n=1)

      if i == depth-2:
        up_skips.append(nn.UpSampling2D((2,2))(encoder_blocks[-1]))
      else:
        up_skips.append(nn.UpSampling2D((2,2))(decoder_blocks[0]))
        decoder_skips.append(nn.UpSampling2D((2**(depth-i-1),2**(depth-i-1)))(encoder_blocks[-1]))

      # Add instead of concat, per half u-net paper
      decoder_blocks.insert(0, nn.Add()(down_skips + up_skips + decoder_skips)) 
      if i == 0:
        # decoder_blocks[0] = self._make_conv_block(nn.Concatenate()([ hull_decoder, decoder_blocks[0] ]), features, n=1)
        decoder_blocks[0] = self._make_conv_block(nn.Concatenate()([ decoder_blocks[0] ]), features, n=1)
        decoder_blocks[0] = self._make_conv_block(decoder_blocks[0], num_output_classes, normalize=False, activation=None, n=1)
      else:
        # hull_link = nn.MaxPool2D((2**i, 2**i))(hull_decoder)
        # decoder_blocks[0] = self._make_conv_block(nn.Concatenate()([ hull_link, decoder_blocks[0] ]), features, n=1)
        decoder_blocks[0] = self._make_conv_block(nn.Concatenate()([ decoder_blocks[0] ]), features, n=1)
        decoder_blocks[0] = self._make_conv_block(decoder_blocks[0], features, n=1)

    decoder_blocks[0] = nn.Activation('softmax', name="Decoder-out-0")(decoder_blocks[0])

    # outs = [ nn.Concatenate(name="Out-0")([ hull_decoder, decoder_blocks[0] ]) ]
    outs = [ decoder_blocks[0] ]
    
    if training:
      for i in range(1,depth-1):
        decoder_blocks[i] = self._make_conv_block(decoder_blocks[i], num_output_classes, normalize=False, activation=None, n=1)
        decoder_blocks[i] = nn.UpSampling2D((2**i, 2**i), interpolation='bilinear')(decoder_blocks[i])
        decoder_blocks[i] = nn.Activation('softmax', name='Decoder-out-{}'.format(i))(decoder_blocks[i])
        # outs.append(nn.Concatenate(name="Out-{}".format(i))([ hull_decoder, decoder_blocks[i] ]))
        outs.append(decoder_blocks[i])
      
      encoder_blocks[-1] = self._make_conv_block(encoder_blocks[-1], num_output_classes, normalize=False, activation=None, n=1)
      encoder_blocks[-1] = nn.UpSampling2D(( 2**(depth-1), 2**(depth-1) ), interpolation='bilinear')(encoder_blocks[-1])
      encoder_blocks[-1] = nn.Activation('softmax', name="Encoder-out")(encoder_blocks[-1])
      # outs.append(nn.Concatenate(name="Out-3")([ hull_decoder, encoder_blocks[-1] ]))
      outs.append(encoder_blocks[-1])

    model = Model(inputs=inputs, outputs=outs)

    return model

  def _make_conv_block(self, x, features, normalize=True, activation='relu', n=2, data_format='channels_last'):
    for i in range(n):
      x = nn.Conv2D(features, (3, 3), activation=None, padding='same', data_format=data_format)(x)
      if n > 1:
        x = nn.Dropout(0.2)(x)
      if normalize:
        x = nn.BatchNormalization()(x)
      if activation is not None:
        x = nn.Activation(activation)(x)

    return x