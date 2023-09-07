import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import plot_vector

def fix_transformation(transformation):
    fixed_transformation = np.zeros((4,4))
    fixed_transformation[3,3] = 1
    q = R.from_matrix(transformation[:3, :3]).as_quat()
    q_mag = np.linalg.norm(q)
    q = q/q_mag
    fixed_rot_mat = R.from_quat(q).as_matrix()
    fixed_transformation[:3, :3] = fixed_rot_mat
    return fixed_transformation

def get_quat_from_matrix(transformation):
    rot_mat = np.zeros((3,3))
    rot_mat = transformation[:3, :3]
    q = R.from_matrix(rot_mat).as_quat()
    q_mag = np.linalg.norm(q)
    q = q/q_mag
    return q

def get_observations(class_id):
    path = "/home/daniel/iiwa_ws/src/handover_orientation_analysis/observations"
    root, dirs, _ = next(os.walk(path))
    for dir in dirs:
        if dir == class_id:
            _, _, files = next(os.walk(os.path.join(root,dir)))
            number_of_files = len(files)
            observations = np.zeros((number_of_files, 4))
            for i in range(0, number_of_files):
                transformation = np.load(os.path.join(root,dir,files[i]))
                fixed_transformation = fix_transformation(transformation)
                observations[i, :] = get_quat_from_matrix(fixed_transformation)
    return observations

def rotate_frame(rotation):
    #print("===ROTATION===\n",rotation)
    rot_mat = R.from_quat(rotation).as_matrix()
    #print("===ROTATION MAT===\n", rot_mat)
    unit_frame = np.eye(3)
    #print("===UNIT FRAME===\n",unit_frame)
    rotated_frame = np.matmul(unit_frame, rot_mat)
    #print("===ROTATED FRAME===\n", rotated_frame)
    return rotated_frame

def get_classes():
    path = "/home/daniel/iiwa_ws/src/handover_orientation_analysis/observations"
    root, dirs, _ = next(os.walk(path))
    return dirs

def plot(ax, vector, col):
    origin = np.array([0.0, 0.0, 0.0])
    dir = np.reshape(vector, 3)
    ax = plot_vector(ax, start = origin, direction = dir, color=col)
    return ax


if __name__ == '__main__':

    classes = get_classes()
    print("===CLASSES===\n", classes)

    for class_id in classes:
        observations = get_observations(class_id)

        figs = [None, None, None]
        ax = [None, None, None]
        axis_limits = range(-1,1)
        for i in range(3):
            if i == 0:
                axis = 'x'
            elif i == 1:
                axis = 'y'
            else:
                axis = 'z'
            plot_title = class_id + "\nOriented " + axis + " axes"
            figs[i] = plt.figure()
            ax[i] = figs[i].add_subplot(projection="3d")
            ax[i].set_title(plot_title, fontsize = 20)
            ax[i].set_xlim(-1, 1)
            ax[i].set_ylim(-1, 1)
            ax[i].set_zlim(-1, 1)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_zticklabels([])

        for i in range(len(observations)):
            rotated_frame = rotate_frame(observations[i])
            ax[0] = plot(ax[0], rotated_frame[:, 0], "red")
            ax[1] = plot(ax[1], rotated_frame[:, 1], "green")
            ax[2] = plot(ax[2], rotated_frame[:, 2], "blue")

        filename = "/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation/axes/" + class_id + "_x.pdf"
        figs[0].savefig(filename, bbox_inches = 'tight', pad_inches = 0)
        filename = "/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation/axes/" + class_id + "_y.pdf"
        figs[1].savefig(filename, bbox_inches = 'tight', pad_inches = 0)
        filename = "/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation/axes/" + class_id + "_z.pdf"
        figs[2].savefig(filename, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
