import streamlit as st
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
import plotly.graph_objects as go
import folium
from streamlit.components.v1 import html
import time
import tempfile

# Set the title of the app
st.set_page_config(page_title="MotoRescue", page_icon="🛣️", layout="wide")

# The rest of your Streamlit app code
st.title("Motorescue - Lane Detection and Sensor Data Visualization")

# Initialize session state for dummy data if not already done
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = {
        "AccX": 0.0,
        "AccY": 0.0,
        "AccZ": 0.0,
        "Tilt": 0.0,
        "Latitude": 0.0,
        "Longitude": 0.0,
        "BuzzerOn": False,
        "MotorCondition": "Stable",
    }


# Function to generate dummy data
def generate_dummy_data():
    # 90% chance to stay in the stable range (1-2 degrees)
    if np.random.rand() > 0.1:
        tilt_angle = np.random.uniform(1, 2)
    else:
        # 10% chance to jump to an unstable range (> 45 degrees)
        tilt_angle = np.random.uniform(45, 50)

    buzzer_on = tilt_angle > 45
    motor_condition = "Unstable" if buzzer_on else "Stable"
    
    return {
        "AccX": np.random.uniform(1, 1.5),
        "AccY": np.random.uniform(1, 1.5),
        "AccZ": np.random.uniform(1, 1.5),
        "Tilt": tilt_angle,
        "Latitude": -7.7787564,
        "Longitude": 110.2328568,
        "BuzzerOn": buzzer_on,
        "MotorCondition": motor_condition,
    }


# Define function to create a gauge chart
def create_gauge_chart(value, min_val, max_val, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={"axis": {"range": [min_val, max_val]}},
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
    return fig


# Function to create and display a Folium map
def display_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], tooltip="UNU Yogyakarta").add_to(m)
    map_html = m._repr_html_()
    html(map_html, height=500)


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
            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                if m < 0:
                    left_lines.append((m, c))
                else:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0) if left_lines else np.array([0, 0])
    right_line = np.mean(right_lines, axis=0) if right_lines else np.array([0, 0])

    rows, cols = image.shape[:2]
    y1 = rows
    y2 = int(rows * 0.6)

    def line_points(slope, intercept):
        if slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    left_points = line_points(*left_line) if left_line.size > 0 else None
    right_points = line_points(*right_line) if right_line.size > 0 else None

    line_img = np.zeros_like(image)
    if left_points:
        draw_lines(line_img, [np.array([left_points])])
    if right_points:
        draw_lines(line_img, [np.array([right_points])])

    if left_points and right_points:
        poly_vertices = [
            (left_points[0], left_points[1]),
            (left_points[2], left_points[3]),
            (right_points[2], right_points[3]),
            (right_points[0], right_points[1]),
        ]
        cv2.fillPoly(img, pts=np.array([poly_vertices], "int32"), color=(0, 255, 0))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.0)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap
    )
    if lines is not None:
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        line_img = slope_lines(line_img, lines)
        return line_img
    else:
        return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
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
    houghed_lines = hough_lines(
        img=masked_img,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        min_line_len=20,
        max_line_gap=180,
    )
    output = weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1.0, γ=0.0)
    return output


def overlay_pipeline(image):
    lane_img = lane_finding_pipeline(image)
    lane_img = cv2.resize(lane_img, (image.shape[1], image.shape[0]))
    return lane_img


# Initialize map and marker in session state if not already done
if "map" not in st.session_state:
    st.session_state.map = folium.Map(
        location=[-7.80, 110.40], zoom_start=12
    )  # Default location
    st.session_state.marker = folium.Marker(
        location=[-7.80, 110.40], tooltip="UNU Yogyakarta"
    )
    st.session_state.marker.add_to(st.session_state.map)

# Initialize state variables for Buzzer Status and Motor Condition
if "last_buzzer_status" not in st.session_state:
    st.session_state.last_buzzer_status = None
if "last_motor_condition" not in st.session_state:
    st.session_state.last_motor_condition = None

# Fixed sidebar
with st.sidebar:
    option = st.selectbox("Choose an option", ["Image", "Video", "Show IoT Data"])
    use_iot_data = st.checkbox("Use IoT Device Data", value=False)

data_placeholder = st.empty()
video_placeholder = st.empty()

if option == "Image":
    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"], key="image_uploader"
    )
    if uploaded_image:
        # Display the original image
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption="Original Image")

        # Process and display the image
        image_np = np.array(original_image)
        result_image = overlay_pipeline(image_np)
        st.image(result_image, caption="Processed Image")

elif option == "Video":
    uploaded_video = st.file_uploader(
        "Choose a video...", type=["mp4"], key="video_uploader"
    )
    if uploaded_video:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name

        # Show preview of the uploaded video
        st.write("Preview of the uploaded video:")
        st.video(temp_file_path)  # Display uploaded video

        # Create a temporary file for the processed video
        processed_video_path = tempfile.mktemp(suffix=".mp4")

        # Function to process video frames and update progress
        def process_video_with_progress(input_path, output_path):
            input_clip = VideoFileClip(input_path)
            total_frames = int(input_clip.fps * input_clip.duration)
            current_frame = 0

            def process_frame(frame):
                nonlocal current_frame
                current_frame += 1
                # Calculate progress and ensure it's within [0.0, 1.0]
                progress = min(current_frame / total_frames, 1.0)
                progress_bar.progress(progress)
                return overlay_pipeline(frame)

            processed_clip = input_clip.fl_image(process_frame)

            # Write processed video file
            processed_clip.write_videofile(
                output_path, codec="libx264", audio=False, threads=4
            )

        # Initialize progress bar
        st.write("Processing video...")
        progress_bar = st.progress(0)

        # Process the video and update progress bar
        process_video_with_progress(temp_file_path, processed_video_path)

        # Display the processed video once processing is complete
        st.write("Processed video:")
        st.video(processed_video_path)

elif option == "Show IoT Data":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig_acc_x_placeholder = st.empty()
    with col2:
        fig_acc_y_placeholder = st.empty()
    with col3:
        fig_acc_z_placeholder = st.empty()
    with col4:
        fig_tilt_placeholder = st.empty()

    # Create placeholders for Buzzer Status and Motor Condition
    with st.container():
        st.markdown("### Sensor Status")
        col_a, col_b = st.columns(2)
        with col_a:
            buzzer_status_placeholder = st.empty()
        with col_b:
            motor_condition_placeholder = st.empty()

    # Display the map based on fixed coordinates
    if "map_displayed" not in st.session_state:
        st.session_state.map_displayed = True
        fixed_lat = -7.7787564
        fixed_lon = 110.2328568
        map_instance = folium.Map(location=[fixed_lat, fixed_lon], zoom_start=15)
        folium.Marker(
            location=[fixed_lat, fixed_lon], tooltip="Sensor Location"
        ).add_to(map_instance)
        map_html = map_instance._repr_html_()
        st.components.v1.html(map_html, height=400)

    while True:
        st.session_state.sensor_data = generate_dummy_data()

        fig_acc_x = create_gauge_chart(
            st.session_state.sensor_data["AccX"], -10, 10, "AccX"
        )
        fig_acc_y = create_gauge_chart(
            st.session_state.sensor_data["AccY"], -10, 10, "AccY"
        )
        fig_acc_z = create_gauge_chart(
            st.session_state.sensor_data["AccZ"], -10, 10, "AccZ"
        )
        fig_tilt = create_gauge_chart(
            st.session_state.sensor_data["Tilt"], -90, 90, "Tilt"
        )

        fig_acc_x_placeholder.plotly_chart(fig_acc_x, use_container_width=True)
        fig_acc_y_placeholder.plotly_chart(fig_acc_y, use_container_width=True)
        fig_acc_z_placeholder.plotly_chart(fig_acc_z, use_container_width=True)
        fig_tilt_placeholder.plotly_chart(fig_tilt, use_container_width=True)

        # Update Buzzer Status and Motor Condition only if they change
        buzzer_status = "On" if st.session_state.sensor_data["BuzzerOn"] else "Off"
        motor_condition = st.session_state.sensor_data["MotorCondition"]

        if st.session_state.last_buzzer_status != buzzer_status:
            buzzer_status_placeholder.metric(label="Buzzer Status", value=buzzer_status)
            st.session_state.last_buzzer_status = buzzer_status

        if st.session_state.last_motor_condition != motor_condition:
            motor_condition_placeholder.metric(
                label="Motor Condition", value=motor_condition
            )
            st.session_state.last_motor_condition = motor_condition

        # Pause for 0.5 seconds before updating
        time.sleep(0.01)
