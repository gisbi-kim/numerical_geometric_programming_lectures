"""
study mateiral recommendation:
    2016 ICRA SLAM tutorial 
    http://www.diag.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_grisetti.pdf
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_rotation_matrix(roll, pitch, yaw, rotmatname="R"):

    #
    # change this values for various experiements 
    init_bias_roll = np.deg2rad(180)
    init_bias_pitch = np.deg2rad(90)
    init_bias_yaw = np.deg2rad(10)
    #

    roll += init_bias_roll
    pitch += init_bias_pitch
    yaw += init_bias_yaw

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0], 
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))

    print(f"{rotmatname}: \n {R}")

    return R

# Generate samples R1, R2, R3
roll1, pitch1, yaw1 = np.deg2rad(12.0), np.deg2rad(3.0), np.deg2rad(1.5)
roll2, pitch2, yaw2 = np.deg2rad(-1.0), np.deg2rad(5.5), np.deg2rad(-2.0)
roll3, pitch3, yaw3 = np.deg2rad(3.5), np.deg2rad(-12.5), np.deg2rad(4.0)

R1 = create_rotation_matrix(roll1, pitch1, yaw1, "R1")
R2 = create_rotation_matrix(roll2, pitch2, yaw2, "R2")
R3 = create_rotation_matrix(roll3, pitch3, yaw3, "R3")


# Initial solution
R0 = np.eye(3)
# R0 = R1.copy()

# rotation matrix to tangent space
def log_map(dR):
    # see https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Log_map_from_SO(3)_to_%F0%9D%94%B0%F0%9D%94%AC(3)
    angleaxis = np.arccos((np.trace(dR) - 1) / 2)
    if np.sin(angleaxis) == 0:
        return np.zeros((3, 3))

    k = (angleaxis / (2 * np.sin(angleaxis)))
    return k * (dR - dR.T)

def unskew(skew_mat):
    assert skew_mat.shape == (3, 3), "Input must be a 3x3 matrix"
    x = skew_mat[2, 1]
    y = skew_mat[0, 2]
    z = skew_mat[1, 0]
    return np.array([x, y, z])

# tangent space to rotation matrix
def exp_map(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    omega_hat = omega / theta
    omega_cross = np.array([[0, -omega_hat[2], omega_hat[1]],
                            [omega_hat[2], 0, -omega_hat[0]],
                            [-omega_hat[1], omega_hat[0], 0]])
    return np.eye(3) + np.sin(theta) * omega_cross + (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)

print(" ")

# Iterative Gauss-Newton opt
max_iters = 300
tolerance = 1e-6

for i in range(max_iters):
    H = np.zeros((3, 3))
    b = np.zeros(3)
    
    for R_i in [R1, R2, R3]:
        e_i = unskew(log_map(np.dot(R0.T, R_i)))  # residual
        J_i = -1*np.eye(3)  # approx. to -I
        """
        Derivation (why -I)

        let 
            R0_adjusted = R0 @ Exp(theta) 

        then, 
            error = unskew(log_map(dR)),
            where dR = R_measured - R_modeled 
                     = R1 - R0_adjusted # same for R2, R3, ...
                     = R0_adjusted.T @ R1 
                     = (R0@Exp(theta)).T @ R1
                     = (R0@(I + skew(theta))).T @ R1
                    
        we can summarize log_map as 
            = k * (dR - dR.T) # k is some constant (not a symbol)
            = k * ( (R0@(I + skew(theta))).T @ R1) - ((R0@(I + skew(theta))).T @ R1).T )
            = k * ( (R0@(I + skew(theta))).T @ R1 - R1.T@(R0@(I + skew(theta))) )
            = k * ( ((I + skew(theta)).T@R0.T@R1) - R1.T@R0@(1 + skew(theta)) )
            we use (I + skew(theta)).T = (I - skew(theta))
            then
            = k * ( ((I - skew(theta))@R0.T@R1) - R1.T@R0@(1 + skew(theta)) )
            = k * ( -1*(R0.T@R1 + R1.T@R0)@skew(theta) + c ) # c is some constant 
            we can assume c ~ 0, and R0.T@R1 = dR ~ I, 
            also, angleaxis = np.arccos((np.trace(dR) - 1) / 2) ~ 0, 
            and k = (angleaxis / (2 * np.sin(angleaxis))) ~ 1/2 
            thus, 
            = 1/2 * (-1*(1 + 1)@skew(theta) + 0) 
            = -skew(theta)

        so unskew it, 
            = -1 * theta 
            
        get derivate of (-1*theta) w.r.t theta is 
            = -I
        """

        H += J_i.T @ J_i 
        b += J_i.T @ e_i 
    
    # solve the normal equation H * dtheta = -b 
    dtheta = np.linalg.solve(H, -b)
    
    print(f"iter {i}: dtheta: {dtheta}")
    if np.linalg.norm(dtheta) < tolerance:
        print(f"The solution converged. terminate the GN opt at iteration {i}\n")
        break
    
    # update R0 
    R0 = R0 @ exp_map(dtheta)

    # check SO(3)ness: make det = 1
    U, _, Vt = np.linalg.svd(R0)
    R0 = np.dot(U, Vt)
    if np.linalg.det(R0) < 0:
        U[:, -1] *= -1
        R0 = np.dot(U, Vt)

R0_star = R0

print("The optimized averaged rotation R0*:")
print(R0_star)


#
# Visualize the reults
# 
def plot_rotation(R, color, label, ax):
    origin = np.zeros(3)
    for i in range(3):
        vec = R[:, i]
        ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], 
                  color=color[i], length=1.0, normalize=True, label=f"{label} Axis" if i == 0 else "")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# frame axis with black color
ax.quiver(0, 0, 0, 1, 0, 0, color='k', linewidth=1, label='Original X')
ax.quiver(0, 0, 0, 0, 1, 0, color='k', linewidth=1, label='Original Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='k', linewidth=1, label='Original Z')

# R1, R2, R3, R0_star 
plot_rotation(R1, ['r', 'r', 'r'], 'R1', ax)
plot_rotation(R2, ['c', 'c', 'c'], 'R2', ax)
plot_rotation(R3, ['orange', 'orange', 'orange'], 'R3', ax)
plot_rotation(R0_star, ['blue', 'blue', 'blue'], 'R0*', ax)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of Rotation Matrices (R1, R2, R3, R0*)')
ax.legend()

plt.show()
