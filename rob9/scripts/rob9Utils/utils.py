import cv2
import numpy as np
from rob9Utils.affordancetools import getPredictedAffordances

def convexHullFromContours(contours):
    """ Input:
        contours    - list [cv2 contours]

        Output:
        hulls       - list [N, cv2 hulls]
    """

    hulls = []
    for contour in contours:
        hulls.append(cv2.convexHull(contour, False))

    return hulls

def maskFromConvexHull(height, width, hulls):
    """ Input:
        height          - int
        width           - int
        hulls           - cv2 hulls

        Output:
        im              - np.array, bool, shape (h, w)
    """

    im = np.zeros((height, width))

    for i in range(len(hulls)):
        cv2.drawContours(im, hulls, i, (255, 255, 255), -1)

    return im.astype(np.bool)

def thresholdMaskBySize(mask, threshold = 0.05):
    """ input:
        mask        - binary mask of shape (h, w)
        threshold   - threshold as total percentage of mask size

        output:
        mask_percentage_of_object - float, percentage size
        keep                      - boolean
    """

    size = mask.shape[0] * mask.shape[1]
    occurences = np.count_nonzero(mask == True)
    mask_percentage_of_object = occurences / size

    keep = True
    if mask_percentage_of_object < threshold:
        keep = False

    return mask_percentage_of_object, keep

def removeOverlapMask(masks):
    """ Sets the pixels of intersection of unions of masks to 0

        Input:
        masks   - np.array, shape (affordances, h, w)

        Output:
        masks   - np.array, shape (affordances, h, w)
    """

    unique_affordances = getPredictedAffordances(masks)
    for i in range(len(unique_affordances)):
        for j in range(i + 1, len(unique_affordances)):
            overlap = masks[unique_affordances[i]] == masks[unique_affordances[j]]
            masks[unique_affordances[i], overlap] = False
            masks[unique_affordances[j], overlap] = False

    return masks


def keepLargestContour(contours):
    """ Returns the largest contour in a list of cv2 contours

        Input:
        contours    - list [cv2 contours]

        Output:
        [contour]   - list with one item
    """

    maxArea = 0
    idx = 0
    for count, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > maxArea:
            maxArea = area
            idx = count

    return [contours[idx]]

def erodeMask(affordance_id, masks, kernel):

    m = masks[affordance_id, :, :]
    m = cv2.erode(m, kernel)
    masks[affordance_id] = m

    return masks
