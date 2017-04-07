"""
Created on Apr 5, 2017

Utility module for dimensionality in convolutions

@author: Levan Tsinadze
"""

def calculate_dimensions(input_height, input_width, filter_height, filter_width, P, S):
  """Calculates output dimesion for convolutional and pooling layers
    Args:
      input_height - input tensor height
      input_width - input tensor width
      filter_height - filter matrix height
      filter_width - filter matrix width
      P - padding
      S - stride
    Returns:
      tuple of -
        new_height - output tensor height
        new_width - output tensor width
  """

  new_height = (input_height - filter_height + 2 * P) / S + 1
  new_width = (input_width - filter_width + 2 * P) / S + 1
 
  return (new_height, new_width) 
