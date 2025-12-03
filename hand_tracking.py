import cv2
import numpy as np
import time
from collections import deque

class HandTrackingSystem:
    def __init__(self):
        self.cap = None
        self.virtual_object = None
        self.state = "SAFE"
        self.state_history = deque(maxlen=10)
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        self.setup_virtual_object()
        
    def setup_virtual_object(self):
        """Create a virtual object (circle) with boundary zones"""
        self.virtual_center = (320, 240)  # Center of screen
        self.virtual_radius = 60  # Radius of virtual object
        self.warning_radius = 120  # Warning zone radius
        self.danger_radius = 80  # Danger zone radius
        
    def preprocess_frame(self, frame):
        """Apply preprocessing for better hand detection"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        return blurred
    
    def detect_hand_contour(self, mask):
        """Find hand contour using skin mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by area to avoid small noise
        area = cv2.contourArea(largest_contour)
        if area < 1000:  # Minimum hand area threshold
            return None
        
        return largest_contour
    
    def get_hand_position(self, contour):
        """Calculate hand position from contour"""
        if contour is None:
            return None
            
        # Get convex hull of the hand
        hull = cv2.convexHull(contour)
        
        # Check if hull is not empty
        if hull is not None and len(hull) > 0:
            # Reshape hull to 2D array
            hull_points = hull.reshape(-1, 2)
            # Find top-most point (minimum y - fingertip approximation)
            if len(hull_points) > 0:
                top_idx = np.argmin(hull_points[:, 1])
                top_point = (int(hull_points[top_idx, 0]), int(hull_points[top_idx, 1]))
                return top_point
        
        # Fallback to contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        
        return None
    
    def calculate_distance_state(self, hand_pos):
        """Calculate distance to virtual object and determine state"""
        if hand_pos is None:
            return "SAFE", 999
        
        # Calculate Euclidean distance from hand to virtual object center
        distance = np.sqrt((hand_pos[0] - self.virtual_center[0])**2 + 
                          (hand_pos[1] - self.virtual_center[1])**2)
        
        # Determine state based on distance
        if distance < self.virtual_radius:
            return "DANGER", distance
        elif distance < self.danger_radius:
            return "DANGER", distance
        elif distance < self.warning_radius:
            return "WARNING", distance
        else:
            return "SAFE", distance
    
    def draw_interface(self, frame, hand_pos, state, distance):
        """Draw all visual elements on the frame"""
        h, w = frame.shape[:2]
        
        # Draw virtual object with zones
        # Warning zone (yellow)
        cv2.circle(frame, self.virtual_center, self.warning_radius, 
                  (0, 255, 255), 2)
        
        # Danger zone (orange)
        cv2.circle(frame, self.virtual_center, self.danger_radius, 
                  (0, 165, 255), 2)
        
        # Virtual object (red)
        cv2.circle(frame, self.virtual_center, self.virtual_radius, 
                  (0, 0, 255), -1)  # Filled circle
        cv2.circle(frame, self.virtual_center, self.virtual_radius, 
                  (255, 255, 255), 2)  # White outline
        
        # Draw hand position if detected
        if hand_pos:
            cv2.circle(frame, hand_pos, 10, (0, 255, 0), -1)
            cv2.circle(frame, hand_pos, 15, (0, 255, 0), 2)
            
            # Draw line from hand to virtual object
            cv2.line(frame, hand_pos, self.virtual_center, (255, 255, 0), 2)
            
            # Draw distance text
            cv2.putText(frame, f"Dist: {int(distance)}px", 
                       (hand_pos[0] + 20, hand_pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw state indicator
        if state == "SAFE":
            color = (0, 255, 0)  # Green
            text = "SAFE"
        elif state == "WARNING":
            color = (0, 255, 255)  # Yellow
            text = "WARNING"
        else:  # DANGER
            color = (0, 0, 255)  # Red
            text = "DANGER DANGER"
        
        # Draw state background
        cv2.rectangle(frame, (10, 10), (300, 80), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), color, 2)
        
        # Draw state text
        cv2.putText(frame, f"STATE: {text}", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        
        # Draw "DANGER DANGER" in big text during danger state
        if state == "DANGER":
            text_size = cv2.getTextSize("DANGER DANGER", cv2.FONT_HERSHEY_DUPLEX, 2, 4)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, "DANGER DANGER", (text_x, 100), 
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, "DANGER DANGER", (text_x, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        # Draw FPS counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Move hand toward red circle", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw zones legend
        cv2.putText(frame, "SAFE: >120px", (w - 200, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "WARNING: 80-120px", (w - 200, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "DANGER: <80px", (w - 200, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop for the hand tracking system"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting Hand Tracking System...")
        print("Instructions:")
        print("1. Move your hand in front of the camera")
        print("2. Try to approach the red circle")
        print("3. System will show SAFE/WARNING/DANGER states")
        print("4. Press 'q' to quit")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                self.fps = frame_count / elapsed_time
                self.fps_history.append(self.fps)
                frame_count = 0
                start_time = time.time()
            
            # Process frame for hand detection
            processed = self.preprocess_frame(frame)
            
            # Detect hand contour
            contour = self.detect_hand_contour(processed)
            
            # Get hand position
            hand_pos = self.get_hand_position(contour)
            
            # Calculate state based on distance
            state, distance = self.calculate_distance_state(hand_pos)
            self.state_history.append(state)
            
            # Draw everything on frame
            frame = self.draw_interface(frame, hand_pos, state, distance)
            
            # Display the frame
            cv2.imshow('Hand Tracking - Virtual Boundary System', frame)
            cv2.imshow('Hand Detection Mask', processed)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"\nPerformance Summary:")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Target FPS: ≥ 8.0")
            print(f"Requirement met: {'✓' if avg_fps >= 8.0 else '✗'}")
        
        print("System shutdown.")

def main():
    """Main function to run the hand tracking system"""
    print("=" * 60)
    print("ARVYAX - Virtual Boundary Hand Tracking System")
    print("=" * 60)
    print("\nThis system demonstrates:")
    print("1. Real-time hand tracking using classical CV")
    print("2. Virtual object with boundary zones")
    print("3. Distance-based state logic (SAFE/WARNING/DANGER)")
    print("4. Visual feedback with state overlay")
    print("5. CPU-only execution with target ≥ 8 FPS")
    print("\nTechniques used:")
    print("- HSV color space for skin detection")
    print("- Morphological operations for noise reduction")
    print("- Contour detection and convex hull")
    print("- Euclidean distance calculation")
    print("- Real-time visual feedback")
    print("=" * 60)
    
    # Create and run the system
    tracker = HandTrackingSystem()
    tracker.run()

if __name__ == "__main__":
    main()