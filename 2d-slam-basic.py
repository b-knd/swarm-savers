import numpy as np
import matplotlib.pyplot as plt

# Initialize particles
num_particles = 100
particles = np.random.rand(num_particles, 3)  # x, y, orientation

# World landmarks
landmarks = np.array([[2, 3], [4, 1], [0, 4]])

# Movement function
def move_particles(particles, delta):
    """
    Move particles based on some motion command.
    """
    # Assume delta = [distance, angle_change]
    distance, angle_change = delta
    new_particles = np.zeros_like(particles)
    
    new_particles[:, 0] = particles[:, 0] + distance * np.cos(particles[:, 2] + angle_change)
    new_particles[:, 1] = particles[:, 1] + distance * np.sin(particles[:, 2] + angle_change)
    new_particles[:, 2] = (particles[:, 2] + angle_change) % (2 * np.pi)
    
    return new_particles

# Measurement function
def update_particles(particles, measurement):
    """
    Update particle weights based on how likely they match the measurement.
    """
    weights = np.ones(num_particles)
    for i, landmark in enumerate(landmarks):
        distance = np.sqrt((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2)
        weights *= np.exp(-(distance - measurement[i])**2 / 2)
    
    weights /= np.sum(weights)  # Normalize
    return weights

# Resample particles based on weights
def resample_particles(particles, weights):
    indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
    new_particles = particles[indices]
    return new_particles

# Simulation loop
for _ in range(10):
    # Move particles
    particles = move_particles(particles, [0.1, 0.05])  # Small motion
    
    # Simulate measurement (distance to each landmark)
    measurements = [np.linalg.norm(particles[0, :2] - lm) for lm in landmarks]  # Assume first particle is "correct"
    
    # Update particles based on measurements
    weights = update_particles(particles, measurements)
    
    # Resample according to weights
    particles = resample_particles(particles, weights)
    
    # Plotting
    plt.scatter(particles[:, 0], particles[:, 1], color='r')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color='b', marker='*')
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.pause(0.1)

plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from orbslam2 import ORB_SLAM2

# # Initialize ORB-SLAM2 system
# slam = ORB_SLAM2('path_to_vocabulary_file', 'path_to_settings_file', ORB_SLAM2.MONOCULAR)

# # Open video file
# cap = cv2.VideoCapture('path_to_your_video.mp4')

# # Check if video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Loop through video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Process the frame using ORB-SLAM2
#     pose = slam.process_image(gray)

#     # If a valid pose is returned, plot it
#     if pose is not None:
#         x, y, z = pose[:3, 3]
#         plt.scatter(x, z, color='r')  # Only plot x and z for 2D view

#     # Display the frame
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

# # Shutdown ORB-SLAM2
# slam.shutdown()

# # Show the final trajectory
# plt.xlabel('X')
# plt.ylabel('Z')
# plt.title('Trajectory')
# plt.show()
