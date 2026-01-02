import cv2
import datetime
import os
import time
from collections import deque
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Configuration
OUTPUT_DIR = "recordings"
CONTINUOUS_CHUNK_DURATION = 30  # Duration of each video chunk in seconds
FPS = 15
CODEC = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec - works without external libraries
PRE_MOTION_BUFFER_SIZE = FPS * 3  # 3 seconds of frames
MOTION_THRESHOLD = 500  # Configurable motion sensitivity
MIN_MOTION_FRAMES = 3  # Minimum consecutive frames to trigger motion
POST_MOTION_DELAY = 2  # Seconds to continue recording after motion stops

# Mode switching configuration
DEMO_MODE = True  # Set to True for quick demo, False for real clock-based switching

if DEMO_MODE:
    # Demo mode: Quick switching for demonstration (2 minutes per mode)
    MODE_DURATION = 30  # 2 minutes per mode
else:
    # Real mode: Clock-based switching (even hours = motion, odd hours = continuous)
    MODE_DURATION = None  # Will use real clock

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_video_writer(filename, frame_width, frame_height):
    """Create and return a video writer object"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    return cv2.VideoWriter(filepath, CODEC, FPS, (frame_width, frame_height))

def get_filename(mode):
    """Generate filename based on mode"""
    now = datetime.datetime.now()
    if mode == "continuous":
        return now.strftime("%Y-%m-%d__%H-%M__CONTINUOUS.avi")
    else:
        return now.strftime("%Y-%m-%d__%H-%M-%S__MOTION.avi")

def upload_to_cloudinary(filepath, filename):
    """Upload video to Cloudinary and delete local file on success"""
    try:
        print(f"[UPLOAD] Uploading {filename} to Cloudinary...")
        
        # Simple upload with folder organization
        response = cloudinary.uploader.upload(
            filepath,
            resource_type="video",
            folder="smart-cctv"
        )
        
        print(f"[UPLOAD] Success! URL: {response['secure_url']}")
        
        # Delete local file after successful upload
        try:
            os.remove(filepath)
            print(f"[CLEANUP] Deleted local file: {filename}")
        except Exception as del_error:
            print(f"[CLEANUP] Warning: Could not delete {filename}: {str(del_error)}")
        
        return response['secure_url']
    
    except Exception as e:
        print(f"[UPLOAD] Failed to upload {filename}: {str(e)}")
        print(f"[CLEANUP] Keeping local file: {filename}")
        return None

def detect_motion(frame, prev_frame, threshold):
    """Simple frame difference motion detection"""
    if prev_frame is None:
        return False
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Compute difference
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels
    motion_pixels = cv2.countNonZero(thresh)
    
    return motion_pixels > threshold

def should_continue_mode(mode_start_time, current_mode):
    """Check if current mode should continue based on DEMO_MODE or clock"""
    if DEMO_MODE:
        # Demo mode: run for MODE_DURATION seconds
        return time.time() - mode_start_time < MODE_DURATION
    else:
        # Real mode: check if hour has changed
        current_hour = datetime.datetime.now().hour
        if current_mode == "continuous":
            # Continuous runs during odd hours
            return current_hour % 2 == 1
        else:
            # Motion runs during even hours
            return current_hour % 2 == 0

def continuous_recording_mode(cap, frame_width, frame_height):
    """Mode 1: Continuous recording with configurable chunks"""
    if DEMO_MODE:
        print(f"[MODE 1] Starting continuous recording mode (demo: {MODE_DURATION}s)")
    else:
        current_hour = datetime.datetime.now().hour
        print(f"[MODE 1] Starting continuous recording mode (until hour {current_hour + 1}:00)")
    
    mode_start_time = time.time()
    
    while should_continue_mode(mode_start_time, "continuous"):
        filename = get_filename("continuous")
        writer = get_video_writer(filename, frame_width, frame_height)
        
        print(f"[CONTINUOUS] Recording chunk: {filename}")
        
        # Calculate exact number of frames needed
        total_frames = CONTINUOUS_CHUNK_DURATION * FPS
        frames_written = 0
        
        while frames_written < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera")
                break
            
            writer.write(frame)
            frames_written += 1
            
            # Control frame rate
            time.sleep(1.0 / FPS)
            
            # Check if mode should end mid-chunk
            if not should_continue_mode(mode_start_time, "continuous"):
                break
        
        writer.release()
        duration = frames_written / FPS
        print(f"[CONTINUOUS] Saved: {filename} ({frames_written} frames, {duration:.1f}s)")
        
        # Upload to Cloudinary
        filepath = os.path.join(OUTPUT_DIR, filename)
        upload_to_cloudinary(filepath, filename)
        
        # Check if we should continue
        if not should_continue_mode(mode_start_time, "continuous"):
            break
    
    print("[MODE 1] Continuous recording mode completed")

def motion_recording_mode(cap, frame_width, frame_height):
    """Mode 2: Motion-based recording with pre-motion buffer"""
    if DEMO_MODE:
        print(f"[MODE 2] Starting motion-based recording mode (demo: {MODE_DURATION}s)")
    else:
        current_hour = datetime.datetime.now().hour
        print(f"[MODE 2] Starting motion-based recording mode (until hour {current_hour + 1}:00)")
    
    mode_start_time = time.time()
    
    # Pre-motion circular buffer
    frame_buffer = deque(maxlen=PRE_MOTION_BUFFER_SIZE)
    prev_frame = None
    writer = None
    recording = False
    motion_frame_count = 0
    no_motion_time = 0
    
    while should_continue_mode(mode_start_time, "motion"):
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera")
            break
        
        # Detect motion
        motion_detected = detect_motion(frame, prev_frame, MOTION_THRESHOLD)
        
        if motion_detected:
            motion_frame_count += 1
        else:
            motion_frame_count = 0
        
        # Add frame to buffer
        frame_buffer.append(frame.copy())
        
        # Start recording if motion detected
        if motion_frame_count >= MIN_MOTION_FRAMES and not recording:
            filename = get_filename("motion")
            writer = get_video_writer(filename, frame_width, frame_height)
            
            print(f"[MOTION] Motion detected! Recording: {filename}")
            
            # Write buffered frames (pre-motion footage)
            for buffered_frame in frame_buffer:
                writer.write(buffered_frame)
            
            recording = True
            no_motion_time = time.time()
        
        # Continue recording if already recording
        if recording:
            writer.write(frame)
            
            # Update no-motion timer
            if motion_detected:
                no_motion_time = time.time()
            
            # Stop recording if no motion for POST_MOTION_DELAY seconds
            if time.time() - no_motion_time > POST_MOTION_DELAY:
                writer.release()
                print(f"[MOTION] Saved: {filename}")
                
                # Upload to Cloudinary
                filepath = os.path.join(OUTPUT_DIR, filename)
                upload_to_cloudinary(filepath, filename)
                
                recording = False
                writer = None
        
        prev_frame = frame.copy()
        
        # Control frame rate
        time.sleep(1.0 / FPS)
    
    # Clean up if still recording
    if recording and writer is not None:
        writer.release()
        print(f"[MOTION] Saved: {filename}")
        
        # Upload to Cloudinary
        filepath = os.path.join(OUTPUT_DIR, filename)
        upload_to_cloudinary(filepath, filename)
    
    print("[MODE 2] Motion-based recording mode completed")

def main():
    """Main function to run the smart CCTV system"""
    print("=== Smart CCTV Recording System ===")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Chunk duration: {CONTINUOUS_CHUNK_DURATION}s ({CONTINUOUS_CHUNK_DURATION * FPS} frames at {FPS} FPS)")
    
    if DEMO_MODE:
        print(f"Mode: DEMO (alternates every {MODE_DURATION}s)")
    else:
        print("Mode: REAL CLOCK (even hours = motion, odd hours = continuous)")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print(f"Recording FPS: {FPS}")
    print(f"Codec: XVID")
    print()
    
    try:
        while True:
            if DEMO_MODE:
                # Demo mode: alternate between modes
                continuous_recording_mode(cap, frame_width, frame_height)
                motion_recording_mode(cap, frame_width, frame_height)
            else:
                # Real mode: check current hour
                current_hour = datetime.datetime.now().hour
                if current_hour % 2 == 1:
                    # Odd hour: continuous recording
                    continuous_recording_mode(cap, frame_width, frame_height)
                else:
                    # Even hour: motion detection
                    motion_recording_mode(cap, frame_width, frame_height)
        
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted by user")
    finally:
        cap.release()
        print("[SHUTDOWN] Camera released. Exiting.")

if __name__ == "__main__":
    main()
