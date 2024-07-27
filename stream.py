import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import serial
import time
from moviepy.editor import VideoFileClip

# Function to read data from the serial port
def read_serial_data(ser):
    try:
        if ser.in_waiting > 0:
            return ser.readline().decode('utf-8').strip()
    except Exception as e:
        return str(e)
    return None

# Function to ensure the connection is active
def check_and_reconnect_serial(ser, port='COM8', baudrate=9600, timeout=1):
    if ser is None or not ser.is_open:
        ser = serial.Serial(port, baudrate, timeout=timeout)
    return ser

# Function to parse sensor data
def parse_sensor_data(data):
    sensor_data = {}
    if data:
        lines = data.split('\n')
        for line in lines:
            if "AccX" in line:
                parts = line.split('\t')
                sensor_data['AccX'] = parts[0].split(':')[1].strip()
                sensor_data['AccY'] = parts[1].split(':')[1].strip()
                sensor_data['AccZ'] = parts[2].split(':')[1].strip()
            elif "Tilt" in line:
                sensor_data['Tilt'] = line.split(':')[1].strip().replace(' deg', '')  # Remove ' deg'
            elif "Latitude" in line:
                sensor_data['Latitude'] = line.split(':')[1].strip()
            elif "Longitude" in line:
                sensor_data['Longitude'] = line.split(':')[1].strip()
    return sensor_data

# Define lane detection pipeline functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 != x2:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                if m < 0:
                    left_lines.append((m, c))
                else:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0) if left_lines else (0, 0)
    right_line = np.mean(right_lines, axis=0) if right_lines else (0, 0)

    for slope, intercept in [left_line, right_line]:
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)
        if slope != 0:  # Check for non-zero slope to avoid division by zero
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            poly_vertices.append((x1, y1))
            poly_vertices.append((x2, y2))
            draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

    if len(poly_vertices) >= 4:
        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 255, 0))
    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    if lines is not None:
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        line_img = slope_lines(line_img, lines)
        return line_img
    else:
        return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.95, rows]
    top_right = [cols * 0.55, rows * 0.6]
    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

def lane_finding_pipeline(image):
    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    masked_img = region_of_interest(img=canny_img, vertices=get_vertices(image))
    houghed_lines = hough_lines(img=masked_img, rho=1, theta=np.pi / 180, threshold=20, min_line_len=20, max_line_gap=180)
    output = weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1., γ=0.)
    return output

def overlay_pipeline(image):
    lane_img = lane_finding_pipeline(image)
    lane_img = cv2.resize(lane_img, (image.shape[1], image.shape[0]))
    combined_img = cv2.addWeighted(image, 0.7, lane_img, 0.3, 0)
    return combined_img

st.title('Lane Detection and IoT Data Streamlit App')

# Sidebar options for input selection
option = st.sidebar.selectbox('Select Input Type', ('Image', 'Video', 'Show IoT Data'))

if option == 'Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        # Apply lane detection pipeline
        output_image = lane_finding_pipeline(image)
        st.image(output_image, caption='Processed Image', use_column_width=True)

elif option == 'Video':
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)

        clip = VideoFileClip(tfile.name)
        
        # Add progress bar
        progress_bar = st.progress(0)
        total_frames = int(clip.fps * clip.duration)
        
        # Initialize session state for frame count
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0

        def process_frame_with_progress(frame):
            st.session_state.frame_count += 1
            progress = min(st.session_state.frame_count / total_frames, 1.0)  # Ensure progress is within [0.0, 1.0]
            progress_bar.progress(progress)
            return overlay_pipeline(frame)

        processed_clip = clip.fl_image(process_frame_with_progress)
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        processed_clip.write_videofile(output_path, audio=False)
        
        st.video(output_path)

elif option == 'Show IoT Data':
    st.title('Real-time IoT Data')
    st.info('Reading data from the connected IoT device...')
    
    iot_data_placeholder = st.empty()  # Create an empty placeholder for IoT data
    warning_placeholder = st.empty()   # Create an empty placeholder for warnings
    status_placeholder = st.empty()    # Create an empty placeholder for motor status
    
    # Buka koneksi serial
    ser = serial.Serial(port='COM8', baudrate=9600, timeout=1)
    
    # Loop pembacaan data serial
    while True:
        ser = check_and_reconnect_serial(ser)  # Pastikan koneksi aktif
        serial_data = read_serial_data(ser)
        if serial_data:
            sensor_data = parse_sensor_data(serial_data)
            
            # Determine motor status
            try:
                tilt = float(sensor_data.get('Tilt', 0))
            except ValueError:
                tilt = 0  # Set to a default value if conversion fails
            
            if abs(tilt) > 45:
                # Show warning message
                warning_placeholder.warning("⚠️ Motor unstable: Tilt angle is out of safe range! ⚠️", icon="⚠️")
                motor_status = "Unstable"
                buzzer_status = "Buzzer ON"
                status_color = "red"
                status_icon = "❌"
            else:
                # Clear warning message
                warning_placeholder.empty()
                motor_status = "Stable"
                buzzer_status = "Buzzer OFF"
                status_color = "green"
                status_icon = "✅"
            
            # Format text with status updates
            formatted_text = f"""
            === Sensor MPU6050 ===\n
            AccX: {sensor_data.get('AccX', 'N/A')} g\n
            AccY: {sensor_data.get('AccY', 'N/A')} g\n
            AccZ: {sensor_data.get('AccZ', 'N/A')} g\n
            Tilt: {sensor_data.get('Tilt', 'N/A')} deg\n
            
            === Data GPS ===\n
            Latitude: {sensor_data.get('Latitude', 'N/A')}\n
            Longitude: {sensor_data.get('Longitude', 'N/A')}\n
            
            === Motor Status ===\n
            Status: {status_icon} <span style="color:{status_color};">{motor_status}</span>\n
            Buzzer: {buzzer_status}
            """
            
            # Update IoT data placeholder with the latest data
            iot_data_placeholder.markdown(formatted_text, unsafe_allow_html=True)  # Use markdown for better formatting
        else:
            iot_data_placeholder.write("No data received yet.")
            warning_placeholder.empty()  # Clear warning if no data is received
            status_placeholder.empty()   # Clear status if no data is received
        
        time.sleep(0.3)  # Delay to reduce data reading frequency