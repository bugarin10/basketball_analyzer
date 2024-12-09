def compute_angles(data):


  mapping = {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
    }
    # Helper function to compute the angle between three points
  def calculate_angle(pointA, pointB, pointC):
      # Extract x and y coordinates, ignoring the third (visibility) value
      A = np.array(pointA[:2])
      B = np.array(pointB[:2])
      C = np.array(pointC[:2])
      # Vectors BA and BC
      BA = A - B
      BC = C - B
      # Calculate the cosine of the angle using the dot product formula
      cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
      # Clip the cosine value to avoid numerical errors outside [-1, 1]
      cos_angle = np.clip(cos_angle, -1.0, 1.0)
      # Return the angle in degrees
      return np.degrees(np.arccos(cos_angle))

  # Indices for the required body parts
  right_shoulder = 12
  right_elbow = 14
  right_wrist = 16
  left_shoulder = 11
  left_elbow = 13
  left_wrist = 15
  right_hip = 24
  right_knee = 26
  right_ankle = 28
  left_hip = 23
  left_knee = 25
  left_ankle = 27
    
  # Compute the angles
  angles = [
      calculate_angle(data[right_shoulder], data[right_elbow], data[right_wrist]),  # Right arm
      calculate_angle(data[left_shoulder], data[left_elbow], data[left_wrist]),    # Left arm
      calculate_angle(data[right_hip], data[right_knee], data[right_ankle]),       # Right leg
      calculate_angle(data[left_hip], data[left_knee], data[left_ankle]),          # Left leg
      calculate_angle(data[right_shoulder], data[right_hip], data[right_knee]),    # Right side
      calculate_angle(data[left_shoulder], data[left_hip], data[left_knee]),       # Left side
  ]
  
  return angles

angles = compute_angles(landmarks_list)

angles