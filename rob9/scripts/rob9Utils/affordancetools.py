import numpy as np
import cv2
import open3d as o3d

def getAffordancePointCloudBasedOnVariance(pcd):
    """ Computes 9 points for each affordance, based on standard deviation,
        present in the point cloud

        Input:
        pcd             - o3d.geometry.PointCloud()

        Output:
        pcd_box        - o3d.geometry.PointCloud()
    """

    affordances, affordance_counts = getPredictedAffordancesInPointCloud(pcd)
    affordance_counts = affordance_counts / np.linalg.norm(affordance_counts)

    out_points, out_colors = [], []
    points = np.asanyarray(pcd.points)
    colors = np.asanyarray(pcd.colors)

    if np.max(colors) <= 1.0:
        colors = colors * 255

    label_colors = getAffordanceColors()

    first = True
    for label_count, label_color in enumerate(label_colors):
        if label_count != 0:
            if affordance_counts[label_count] > 0.005:

                idx = colors == label_color
                idx = np.sum(idx, axis = -1) == 3

                if True in idx:
                    aff_points = points[idx]

                    x_c, y_c, z_c = np.mean(aff_points, axis = 0)
                    x_std, y_std, z_std = np.std(aff_points, axis = 0) / 2

                    box_points = []
                    box_points.append([x_c, y_c, z_c]) # centroid
                    box_points.append([x_c - x_std, y_c - y_std, z_c - z_std])
                    box_points.append([x_c + x_std, y_c - y_std, z_c - z_std])
                    box_points.append([x_c - x_std, y_c + y_std, z_c - z_std])
                    box_points.append([x_c + x_std, y_c + y_std, z_c - z_std])
                    box_points.append([x_c - x_std, y_c - y_std, z_c + z_std])
                    box_points.append([x_c + x_std, y_c - y_std, z_c + z_std])
                    box_points.append([x_c - x_std, y_c + y_std, z_c + z_std])
                    box_points.append([x_c + x_std, y_c + y_std, z_c + z_std])

                    box_colors = [label_color for i in range(len(box_points))]

                    box_points = np.array(box_points)
                    box_colors = np.array(box_colors) / 255

                    if first:
                        out_points = box_points
                        out_colors = box_colors
                        first = False
                    else:
                        out_points = np.vstack((out_points, box_points))
                        out_colors = np.vstack((out_colors, box_colors))

    pcd_box = o3d.geometry.PointCloud()
    pcd_box.points = o3d.utility.Vector3dVector(out_points)
    pcd_box.colors = o3d.utility.Vector3dVector(out_colors)

    return pcd_box

def getPredictedAffordancesInPointCloud(pcd):
    """ Input:
        pcd         - o3d.geometry.PointCloud()

        Output:
        affordances - list[], one-hot-encoded vector with present affordances.
        counts      - np.array(), shape(num_affordances), int count of every affordance.
    """

    label_colors = getAffordanceColors()

    colors = np.asanyarray(pcd.colors)
    if np.max(colors) <= 1.0:
        colors = colors * 255

    affordances, counts = [], []

    for label_color in label_colors:
        idx = colors == label_color
        idx = np.sum(idx, axis = -1) == 3

        if True in idx:
            affordances.append(1)
            counts.append(np.count_nonzero(idx == True))
        else:
            affordances.append(0)
            counts.append(0)

    return affordances, np.array(counts)


def getPredictedAffordances(masks, bbox = None):
    """ Input:
        masks       - np.array, shape (affordances, h, w)
        bbox        - if provided speeds up computations

        Output:
        affordances - list, [N, ??], affordances found in each N object
    """

    affordances = []

    if bbox is not None:
        masks = masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    for count, affordance_mask in enumerate(masks):
        if True in np.unique(affordance_mask):
            affordances.append(count)

    return affordances

def getAffordanceColors():
    """ Output:
        colors  -   'official' list of colors so they are uniform
    """

    colors = [(0,0,0), (0,0,255), (0,255,0), (123, 255, 123), (255, 0, 0),
                (255, 255, 0), (255, 255, 255), (255, 0, 255), (123, 123, 123), (255, 255, 0), (70, 70, 70), (0,10,0)]

    return colors

def getAffordanceContours(affordance_id, masks, bbox = None):
    """ Input:
        affordance_id   - int,
        masks           - np.array, bool, shape (affordances, h, w)
        bbox            - np.array, shape (4, 2), if provided speeds up computation

        Output:
        contours        - list[N, cv2 contours]
    """

    contours = []
    m = masks[affordance_id]
    if bbox is not None:
        m = masks[affordance_id, bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.uint8)

    contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def getAffordanceBoundingBoxes(pcd):
    """ Returns 8 points of the non-axis aligned bounding box for each affordance
        present in the provided affordance point cloud

        Input:
        pcd     - o3d.geometry.PointCloud where each point has an affordance
                  assigned.
        Output:
        points  - np.array (N, 3) x, y, z points
        colors  - np.array (N, 3) r, g, b associated with each affordance
    """

    pcd_points = np.asanyarray(pcd.points)
    pcd_colors = np.asanyarray(pcd.colors).astype(np.uint8)

    if np.max(pcd_colors) <= 1.0:
        pcd_colors = pcd_colors * 255

    label_colors = getAffordanceColors()

    points = []
    colors = []

    #print(np.unique(pcd_colors, axis = 0))

    for color in label_colors:
        idx = pcd_colors == color
        idx = np.sum(idx, axis = -1) == 3

        if True in idx:
            aff_points = o3d.utility.Vector3dVector(pcd_points[idx])
            bbox = np.asanyarray(o3d.geometry.OrientedBoundingBox.create_from_points(aff_points).get_box_points())
            bbox_colors = [color for i in range(bbox.shape[0])]

            #print(bbox)

            if len(points) == 0:
                points = bbox
                colors = np.array(bbox_colors)
            else:
                points = np.vstack((points, bbox))
                colors = np.vstack((colors, bbox_colors))

    return points, colors

def getPointCloudAffordanceMask(affordance_id, points, uvs, masks, remove_outliers = True, bbox = None):
    """ Input:
        affordance_id   -   int,
        points          -   np.array, shape: (n, 3), xyz points
        masks           - np.array, boolean, shape (affordances, h, w)
        uvs             - np.array, shape (n, 2)
        remove_outliers - bool, uses open3d remove_statistical_outlier method
        bbox            - np.array, shape (4, 2) if provided speeds up computation

        Output:
        success         - bool, set to False if computation fails
        points          - np.array, shape (n, 3), xyz points
    """

    # Check if points are empty
    if points.shape[0] <= 0:
        return False, 0

    # select affordance mask in bbox of object
    if bbox is not None:
        m = masks[affordance_id, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # do the same for points and uv
        points = points[np.where(uvs[:,0] > bbox[1])]
        uvs = uvs[np.where(uvs[:,0] > bbox[1])]
        points = points[np.where(uvs[:,0] < bbox[3])]
        uvs = uvs[np.where(uvs[:,0] < bbox[3])]
        points = points[np.where(uvs[:,1] > bbox[0])]
        uvs = uvs[np.where(uvs[:,1] > bbox[0])]
        points = points[np.where(uvs[:,1] < bbox[2])]
        uvs = uvs[np.where(uvs[:,1] < bbox[2])]
        uvs[:,0] -= bbox[1]
        uvs[:,1] -= bbox[0]

    else:
        m = masks[affordance_id, :, :]

    # Check if the affordance mask has the requested affordance
    if not 255 in np.unique(m) and not 1 in np.unique(m):
        return False, 0

    # get points belonging to affordance
    cloud_masked = []
    for count, uv in enumerate(uvs):
        if m[uv[0], uv[1]] != False:
            cloud_masked.append(points[count])

    points = np.array(cloud_masked)

    if remove_outliers:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        points = np.asanyarray(pcd.points)

    return True, points

def getObjectAffordancePointCloud(pcd, masks, uvs):
    """ Returns the points in the point cloud associated with the mask affordances
        also correctly collored

        Input:
        pcd     - open3d.geometry.PointCloud()
        masks   - np.array, boolean, shape (affordances, h, w)
        uvs     - np.array, shape (n, 2)

        Output:
        pcd_affordance  - open3d.geometry.PointCloud()
    """
    if masks is None:
        print("No masks available to visualize")
        return 0
    if masks.shape[0] <= 0:
        print("No masks available to visualize")
        return 0

    points = np.asanyarray(pcd.points)

    if points.shape[0] <= 0:
        print("No points in point cloud")
        return 0

    colors = getAffordanceColors()
    #colors = [(0,0,0), (0,0,255), (0,255,0), (123, 255, 123), (255, 0, 0),
    #            (255, 255, 0), (255, 255, 255), (255, 0, 255), (123, 123, 123), (0, 255, 255), (70, 70, 70), (0,10,0)]

    for count, c in enumerate(colors):
        b, g, r = c
        cs = (r / 255.0, g / 255.0, b / 255.0)
        colors[count] = cs

    observed_affordances = getPredictedAffordances(masks)

    aff_points = []
    aff_colors = []
    for aff in observed_affordances:
        _, local_aff_mask = getPointCloudAffordanceMask(affordance_id = aff,
                                        points = points, uvs = uvs, masks = masks)
        local_aff_colors = [colors[aff] for i in range(local_aff_mask.shape[0])]
        local_aff_colors = np.array(local_aff_colors)
        if len(aff_points) == 0:
            aff_points = local_aff_mask
            aff_colors = local_aff_colors
        else:
            aff_points = np.vstack((aff_points, local_aff_mask))
            aff_colors = np.vstack((aff_colors, local_aff_colors))

    pcd_affordance = o3d.geometry.PointCloud()
    pcd_affordance.points = o3d.utility.Vector3dVector(aff_points)
    pcd_affordance.colors = o3d.utility.Vector3dVector(aff_colors)

    return pcd_affordance
