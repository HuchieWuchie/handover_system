import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform

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

def get_the_mean_transformation(class_name, lines):
    mean_orientation = np.zeros(4)
    for line in lines:
        line = line.replace("[","")
        line = line.replace("]","")
        temp = line.split()
        if class_name == temp[0]:
            print(temp)
            mean_orientation = temp[1:5]
            for i in range(0,4):
                mean_orientation[i] = float(mean_orientation[i])
            print(mean_orientation)

    transformation = np.zeros((4,4))
    transformation[3,3] = 1
    rot_mat = R.from_quat(mean_orientation).as_matrix()
    transformation[:3,:3] = rot_mat
    return transformation


if __name__ == '__main__':
    with open('/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation/mean_handover_orientations.txt') as f:
        lines = f.readlines()

    path = "/home/daniel/iiwa_ws/src/handover_orientation_analysis/observations"
    output_path = "/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation"
    root, dirs, _ = next(os.walk(path))
    for dir in dirs:
        _, _, files = next(os.walk(os.path.join(root,dir)))
        number_of_files = len(files)
        ax = None
        for i in range(0, number_of_files):
            transformation = np.load(os.path.join(root,dir,files[i]))
            fixed_transformation = fix_transformation(transformation)
            ax = plot_transform(ax, fixed_transformation, **{'linestyle':'--'})

        ax = plot_transform(ax, get_the_mean_transformation(dir, lines), **{'lw':8})

        plt.title(dir, fontsize=40)
        plt.tight_layout()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.xaxis.label.set_size(30)
        ax.yaxis.label.set_size(30)
        ax.zaxis.label.set_size(30)

        image_path = os.path.join(output_path, 'figures', dir + '.pdf')
        plt.savefig(image_path)
        plt.show()
