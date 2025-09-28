import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit app title
st.title("Core-Shell Nanoparticle Analysis with Coordinate System (Color TEM)")

# File uploader for TEM images
uploaded_files = st.file_uploader("Upload TEM Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Check if at least one image is uploaded
if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(np.array(image))
    
    # Display uploaded images
    st.subheader("Uploaded Images")
    for i, img in enumerate(images):
        st.image(img, caption=f"Image {i+1}", use_column_width=True)

    # Interactive scale bar calibration
    st.subheader("Scale Bar Calibration")
    scale_bar_nm = st.number_input("Enter scale bar length (nm):", min_value=1.0, value=20.0)
    scale_bar_coords = {}
    scale_factors = {}

    for i, img in enumerate(images):
        st.write(f"Click two points on the scale bar for Image {i+1}")
        img_pil = Image.fromarray(img)
        coords = [None, None]
        def get_coords(x, y):
            if coords[0] is None:
                coords[0] = (x, y)
            elif coords[1] is None:
                coords[1] = (x, y)
            scale_bar_coords[i] = coords
        st.image(img, use_column_width=True, output="canvas", on_click=get_coords)
        if st.button(f"Confirm Scale Bar Points for Image {i+1}"):
            if scale_bar_coords.get(i) and None not in scale_bar_coords[i]:
                x1, y1 = scale_bar_coords[i][0]
                x2, y2 = scale_bar_coords[i][1]
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if pixel_distance > 0:
                    scale_factors[i] = scale_bar_nm / pixel_distance
                    st.success(f"Scale factor for Image {i+1}: {scale_factors[i]:.2f} nm/pixel")
                else:
                    st.error("Invalid scale bar points: zero distance detected.")
            else:
                st.error(f"Please click two distinct points for Image {i+1}.")

    # Segmentation parameters
    st.subheader("Segmentation Parameters")
    red_threshold = st.slider("Red Cu Core Threshold (0-255)", 0, 255, (150, 255), key="red")
    green_threshold = st.slider("Green Ag Shell Threshold (0-255)", 0, 255, (50, 150), key="green")
    min_particle_area = st.number_input("Minimum Particle Area (pixels):", min_value=10, value=100)

    # Process images button
    if st.button("Analyze Images"):
        for idx, img in enumerate(images):
            if idx not in scale_factors or scale_factors[idx] is None:
                st.write(f"Skipping Image {idx+1}: Scale bar not calibrated.")
                continue

            st.write(f"Processing Image {idx+1}...")
            scale_factor = scale_factors[idx]
            x1, y1 = scale_bar_coords[idx][0]
            x2, y2 = scale_bar_coords[idx][1]
            height, width = img.shape[:2]

            # Calculate corner coordinates in nm
            top_left_nm = (0, height * scale_factor)
            top_right_nm = (width * scale_factor, height * scale_factor)
            bottom_right_nm = (width * scale_factor, 0)

            st.write(f"Coordinate System (nm) for Image {idx+1}:")
            st.write(f"Bottom-Left: (0, 0)")
            st.write(f"Top-Left: ({top_left_nm[0]:.2f}, {top_left_nm[1]:.2f})")
            st.write(f"Top-Right: ({top_right_nm[0]:.2f}, {top_right_nm[1]:.2f})")
            st.write(f"Bottom-Right: ({bottom_right_nm[0]:.2f}, {bottom_right_nm[1]:.2f})")

            # Convert to HSV for better color segmentation
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Define color ranges for red (Cu core) and green (Ag shell)
            lower_red = np.array([0, 100, red_threshold[0]])
            upper_red = np.array([10, 255, red_threshold[1]])
            lower_green = np.array([40, 50, green_threshold[0]])
            upper_green = np.array([80, 255, green_threshold[1]])

            # Create masks
            red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

            # Morphological operations to clean masks
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and pair contours
            core_radii = []
            shell_radii = []
            shell_thicknesses = []
            img_with_contours = img.copy()

            for red_cnt in red_contours:
                if cv2.contourArea(red_cnt) > min_particle_area:
                    (x, y), core_radius_pixels = cv2.minEnclosingCircle(red_cnt)
                    core_radius_nm = core_radius_pixels * scale_factor
                    core_radii.append(core_radius_nm)

                    for green_cnt in green_contours:
                        if cv2.contourArea(green_cnt) > min_particle_area:
                            (x_shell, y_shell), shell_radius_pixels = cv2.minEnclosingCircle(green_cnt)
                            if abs(x - x_shell) < 10 and abs(y - y_shell) < 10 and shell_radius_pixels > core_radius_pixels:
                                shell_radius_nm = shell_radius_pixels * scale_factor
                                shell_thickness_nm = (shell_radius_pixels - core_radius_pixels) * scale_factor
                                shell_radii.append(shell_radius_nm)
                                shell_thicknesses.append(shell_thickness_nm)
                                cv2.drawContours(img_with_contours, [red_cnt], -1, (0, 0, 255), 1)  # Red Cu core
                                cv2.drawContours(img_with_contours, [green_cnt], -1, (0, 255, 0), 1)  # Green Ag shell
                                break

            # Display image with contours
            st.image(img_with_contours, caption=f"Image {idx+1}: Red Cu Core, Green Ag Shell", use_column_width=True)

            # Calculate and display averages
            avg_core_radius = np.mean(core_radii) if core_radii else 0
            avg_shell_radius = np.mean(shell_radii) if shell_radii else 0
            avg_shell_thickness = np.mean(shell_thicknesses) if shell_thicknesses else 0
            st.write(f"Average Cu Core Radius: {avg_core_radius:.2f} nm")
            st.write(f"Average Ag Shell Radius: {avg_shell_radius:.2f} nm")
            st.write(f"Average Shell Thickness: {avg_shell_thickness:.2f} nm")

            # Plot histograms
            if core_radii:
                fig, ax = plt.subplots()
                ax.hist(core_radii, bins=20, color="red", alpha=0.7)
                ax.set_xlabel("Cu Core Radius (nm)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            if shell_radii:
                fig, ax = plt.subplots()
                ax.hist(shell_radii, bins=20, color="green", alpha=0.7)
                ax.set_xlabel("Ag Shell Radius (nm)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            if shell_thicknesses:
                fig, ax = plt.subplots()
                ax.hist(shell_thicknesses, bins=20, color="purple", alpha=0.7)
                ax.set_xlabel("Shell Thickness (nm)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            if not core_radii and not shell_radii:
                st.warning(f"No valid particles detected in Image {idx+1}. Adjust thresholds or check image.")
