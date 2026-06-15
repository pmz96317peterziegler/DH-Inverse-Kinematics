import numpy as np

# DH parameters of a UR5 robot arm
#
DH_PARAMS = np.array(
    [
        [0.0, np.pi / 2, 0.089159, 0.0],  # Joint 1
        [-0.425, 0.0, 0.0, 0.0],  # Joint 2
        [-0.39225, 0.0, 0.0, 0.0],  # Joint 3
        [0.0, np.pi / 2, 0.10915, 0.0],  # Joint 4
        [0.0, -np.pi / 2, 0.09465, 0.0],  # Joint 5
        [0.0, 0.0, 0.0823, 0.0],  # Joint 6
    ]
)


def dh_transform(a, alpha, d, theta):

    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics(dh_params):

    transforms = []
    T_end_effector_fk = np.eye(4)
    for i in range(dh_params.shape[0]):

        a, alpha, d, theta = dh_params[i]
        T_i = dh_transform(a, alpha, d, theta)
        T_end_effector_fk = T_end_effector_fk @ T_i
        transforms.append(T_i.copy())
        ## print(f"Transformation matrix for joint {i+1}:\n{T_i}\n")
    return T_end_effector_fk, transforms


def jacobian(DH_current):

    # the range is DH_PARAMS + 1 because we are trying to get the frames out of the joint and there are
    # always n_joints + 1 frames (including the end effector)
    # DH_current is just the DH table but with the current joint angles instead of the zeroes in the last column
    Ts = []
    for i in range(DH_current.shape[0] + 1):
        T_0_i = forward_kinematics(DH_current[0:i])[0]
        Ts.append(T_0_i)

    # this part pulls out all of the position vectors for each frame and spits out the final
    # end effector config
    dists = []
    for i in range(DH_current.shape[0] + 1):
        dists.append(Ts[i][0:3, 3])
    end_effector_position = dists[DH_current.shape[0]]

    Jacobian = np.zeros((6, 6))
    z_initial = np.array([0, 0, 1])
    for i in range(6):
        # the rotation matrix for the i'th frame is the top left 3x3 block of the transformation matrix for that frame
        Rotation_0_i = Ts[i][0:3, 0:3]
        z_axis = Rotation_0_i @ z_initial
        # top three rows for any column of the jacobian are filled by the cross product of the z axis of the frame and the vector from that frame to the end effector
        Jacobian[0:3, i] = np.cross(z_axis, end_effector_position - dists[i])
        # the bottom three rows for any column of the jacobian are filled by the z axis of the frame
        Jacobian[3:6, i] = z_axis
    return Jacobian


def orientation_error(R_current, R_target):
    # this function is necessary to find the correct orientation of the end effector
    # however it is not as simple as just subtracting the two rotation matrices
    # we have to find the angle between the two rotation matrices and then find the axis of rotation that would get us from the current orientation to the target orientation
    R_err = R_target @ R_current.T

    # trace is the sum of the diagonal elements of a matrix
    # utilize the equation trace(R) = 1 + 2·cos(θ) to find the angle between the two rotation matrices
    # cos_angle will fall between -1 and 1
    #    1 means perfectly aligned
    #    0 means 90 degrees apart
    #   -1 means 180 degrees apart or exact opposites
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    # these two if statements are just to clean up the numbers to make sure that they can be cleanly
    # utilized in the arccos function and to avoid any numerical issues that might arise
    if cos_angle > 1.0:
        cos_angle = 1.0
    if cos_angle < -1.0:
        cos_angle = -1.0
    angle = np.arccos(cos_angle)

    # if the angle is within a certain tolerance then just count it as the same as the
    # desired angle
    if abs(angle) < 1e-6:
        return np.zeros(3)

    # figures out which axis the rotation is around
    # which turns it into a vector with 3 components which is what we need
    sin_angle = np.sin(angle)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    ) / (2.0 * sin_angle)

    return angle * axis


def inverse_kinematics(
    T_target, Theta_list_init, max_iters=150, tol=1e-4, step=1
):
    """
    Numerical IK.
    Matches both position and orientation of T_target.
    """
    Theta_list = np.array(Theta_list_init, dtype=float).copy()

    target_position = T_target[0:3, 3]
    target_rotation = T_target[0:3, 0:3]

    for iteration in range(max_iters):
        # Build a DH table with current joint angles in column 3
        DH_current = DH_PARAMS.copy()
        for i in range(DH_PARAMS.shape[0]):
            DH_current[i, 3] = Theta_list[i]

        T_current = forward_kinematics(DH_current)[0]
        current_position = T_current[0:3, 3]
        current_rotation = T_current[0:3, 0:3]

        e_pos = target_position - current_position
        e_rot = orientation_error(current_rotation, target_rotation)
        # combines the position and orientation 3D vectors into a single 6D error vector that we can use to update our joint angles
        error = np.concatenate([e_pos, e_rot])
        error_norm = np.linalg.norm(error)

        print("iter %3d   error = %.6f" % (iteration, error_norm))

        if error_norm < tol:
            Theta_list_wrapped = ((Theta_list + np.pi) % (2 * np.pi)) - np.pi
            print("Converged!")
            
            return Theta_list_wrapped

        Jacobian = jacobian(DH_current)

        jacobian_inv = np.linalg.pinv(Jacobian)

        delta_joint_angles = jacobian_inv @ error

        # relies on newton raphson method to update the next set of joint angles
        Theta_list = Theta_list + step * delta_joint_angles

    print("Did not converge")
    return Theta_list


if __name__ == "__main__":
    T_target = np.array(
        [[0, 1, 0, 0.5], [0, 0, -1, -0.1], [-1, 0, 0, 0.1], [0, 0, 0, 1]]
    )
    initial_guess = [-0.27, 4.794, -2.026, -2.634, 3.241, -1.418]

    solution = inverse_kinematics(T_target, initial_guess)
    print("\nSolved joint angles (rad):")
    print(solution)