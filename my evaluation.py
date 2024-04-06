import numpy as np

def compute_iou(box1, box2):
    """Compute the Intersection Over Union of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def precision_recall_at_iou_threshold(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """Calculate precision and recall at a given IoU threshold."""
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_boxes)

    for pred_box in predicted_boxes:
        match_found = False
        for gt_box in ground_truth_boxes:
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
                match_found = True
                break  # Assume each gt box can only match with one predicted box
        if not match_found:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall


def evaluate_model(model, dataset, iou_threshold=0.5):
    all_precisions = []
    all_recalls = []

    for images, labels in dataset:
        probabilities, boxes = model.predict(images)
        # Process model outputs here: convert probabilities to binary indicators, extract bounding boxes, etc.
        # This depends on your model output format and how you've decided to threshold/interpret probabilities

        for i in range(len(images)):
            pred_boxes = extract_boxes(boxes[i], probabilities[i])  # You'll need to implement this based on your model output structure
            gt_boxes = labels['x_boxes_reshape'][i].numpy()  # Adjust based on actual label structure
            precision, recall = precision_recall_at_iou_threshold(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
            all_precisions.append(precision)
            all_recalls.append(recall)

    # Compute overall metrics
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    print(f"Mean Precision: {mean_precision}, Mean Recall: {mean_recall}")
