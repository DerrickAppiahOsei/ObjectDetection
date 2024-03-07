# ObjectDetection
Run the bounding box ...... copy notebook.
TrainingData5zeroes.h5 is the dataset including 10000 images and bounding boxes.
model5zeroes.h5 is the saved model after 3000epochs. 

Data Loading:
    """ Images and Bounding boxes are loaded as lists and keyed to match each other as in arr_98 in images matches arr_98 in boxes. I will attach an h5 file with keys Images and Boxes just to avoid confusion."""
Data Normalization:
    """boxes are normalized as such to keep the first column unaffected since the first column only represents the class. The boxes are normalized to have the same scale as the images thus between zero and one to enable the model to better train."""
The boxes are sliced into probabilities and bounding boxes(boxes_np) and passed as dictionary to the model.Thus two outputs are expected: the probability that a bounding box exists AND the predicted bounding box.
The data is shuffled for better training and generalizability of the model.
