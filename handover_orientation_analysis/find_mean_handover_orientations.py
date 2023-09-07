import os
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation as R

def get_quat_from_matrix(transformation):
    #print(transformation)
    rot_mat = np.zeros((3,3))
    rot_mat = transformation[:3, :3]
    #print(rot_mat)
    q = R.from_matrix(rot_mat).as_quat()
    q_mag = np.linalg.norm(q)
    q = q/q_mag
    return q

def get_random_quaternion():
    q = np.zeros(4)
    q[0] = np.random.uniform(low=-1.0, high=1.0) #q_x
    q[1] = np.random.uniform(low=-1.0, high=1.0) #q_y
    q[2] = np.random.uniform(low=-1.0, high=1.0) #q_z
    q[3] = np.random.uniform(low=-1.0, high=1.0) #q_w
    q_mag = np.linalg.norm(q)
    q = q/q_mag
    return q

def get_distance(s, r):
    return min( np.linalg.norm(s-r), np.linalg.norm(s+r) )

def objective_function(solution, observations):
    distances = []
    for observation in observations:
        distance = get_distance(solution, observation)
        distances.append(distance)
    distances = np.asarray(distances)
    return distances.sum()

def find_solution(observations):
    min_sum = None
    min_sum_solution = None
    #print("===MINIMIZATION===")
    for i in range(0, 50):
        #print("===ITERATION " + str(i) + " ===")
        #print("===INITIAL GUESS===")
        initial_guess = get_random_quaternion()
        #print(initial_guess)
        xopt = scipy.optimize.minimize(objective_function, x0 = initial_guess, method='Nelder-Mead', args=(observations, ))
        # PRINT ALL THE OUTPUT
        #print(xopt)
        # PRINT ONLY THE SOLUTION
        #print(xopt.x)

        # NORMALIZE THE SOLUTION
        solution_mag = np.linalg.norm(xopt.x)
        solution = xopt.x/solution_mag
        #print("===SOLUTION===")
        #print(solution)
        if i == 0:
            min_sum_solution = solution
            min_sum = objective_function(solution, observations)
        else:
            if np.array_equal(min_sum_solution, solution):
                continue
            else:
                sum = objective_function(solution, observations)
                if sum < min_sum:
                    min_sum_solution = solution
                    min_sum = sum
                else:
                    continue

    return min_sum_solution

if __name__ == '__main__':
    path = "/home/daniel/iiwa_ws/src/handover_orientation_analysis/observations"
    output_path = "/home/daniel/iiwa_ws/src/ROB10/mean_handover_orientation/mean_handover_orientations.txt"
    with open(output_path, 'w') as f:
        f.truncate()

    root, dirs, _ = next(os.walk(path))
    for dir in dirs:
        _, _, files = next(os.walk(os.path.join(root,dir)))
        number_of_files = len(files)
        observations = np.zeros((number_of_files, 4))
        for i in range(0, number_of_files):
            transformation = np.load(os.path.join(root,dir,files[i]))
            observations[i, :] = get_quat_from_matrix(transformation)

        solution = find_solution(observations)
        print(solution)

        with open(output_path, 'a') as f:
            f.write(dir + " " + str(solution) + "\n")
