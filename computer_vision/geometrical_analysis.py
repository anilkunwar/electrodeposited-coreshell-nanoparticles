import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit app title
st.title("Core-Shell Nanoparticle Analysis with Coordinate System and Measurement Tools (Color TEM)")

# File uploader for TEM images
uploaded_files = st.file_uploader("Upload TEM Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Check if at least one image is uploaded
if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(np.array(image))
    
    # Display uploaded images with pixel scales (GIMP-like boundary scales)
    st.subheader("Uploaded Images with Pixel Scales")
    for i, img in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels from top)")
        # Set ticks for rulers
        ax.set_xticks(np.arange(0, img.shape[1], step=max(1, img.shape[1]//10)))
        ax.set_yticks(np.arange(0, img.shape[0], step=max(1, img.shape[0]//10)))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
        st.caption(f"Image {i+1} - Size: {img.shape[1]} x {img.shape[0]} pixels")

    # Interactive scale bar calibration using S1 and S2
    st.subheader("Scale Bar Calibration")
    scale_bar_coords = {}
    scale_factors = {}

    for i, img in enumerate(images):
        st.write(f"Enter pixel coordinates for S1 and S2 (ends of the scale bar) for Image {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            s1_x = st.number_input(f"S1 X for Image {i+1}", min_value=0, max_value=img.shape[1]-1, value=0, key=f"s1_x_{i}")
            s1_y = st.number_input(f"S1 Y for Image {i+1}", min_value=0, max_value=img.shape[0]-1, value=0, key=f"s1_y_{i}")
        with col2:
            s2_x = st.number_input(f"S2 X for Image {i+1}", min_value=0, max_value=img.shape[1]-1, value=img.shape[1]//2, key=f"s2_x_{i}")
            s2_y = st.number_input(f"S2 Y for Image {i+1}", min_value=0, max_value=img.shape[0]-1, value=img.shape[0]//2, key=f"s2_y_{i}")
        
        scale_bar_nm = st.number_input(f"Scale bar length (nm) for Image {i+1}", min_value=1.0, value=20.0, key=f"nm_{i}")
        
        if st.button(f"Confirm Scale Bar Points for Image {i+1}"):
            scale_bar_coords[i] = [(s1_x, s1_y), (s2_x, s2_y)]
            pixel_distance = np.sqrt((s2_x - s1_x)**2 + (s2_y - s1_y)**2)
            if pixel_distance > 0:
                scale_factors[i] = scale_bar_nm / pixel_distance
                st.success(f"Scale factor for Image {i+1}: {scale_factors[i]:.2f} nm/pixel")
            else:
                st.error("Invalid scale bar points: zero distance detected.")

    # Interactive Measurement Tool with Sliders for Two Points
    st.subheader("Interactive Measurement Tool")
    selected_image_idx = st.selectbox("Select Image for Measurement", options=range(len(images)), format_func=lambda x: f"Image {x+1}")
    if selected_image_idx is not None:
        img = images[selected_image_idx]
        height, width = img.shape[:2]
        
        # Define fixed corner points B1, B2, B3, B4 (top-left as (0,0))
        b1 = (0, 0)  # Top-left
        b2 = (width - 1, 0)  # Top-right
        b3 = (width - 1, height - 1)  # Bottom-right
        b4 = (0, height - 1)  # Bottom-left
        
        st.write("Fixed Corner Coordinates (pixels):")
        st.write(f"B1 (Top-Left): {b1}")
        st.write(f"B2 (Top-Right): {b2}")
        st.write(f"B3 (Bottom-Right): {b3}")
        st.write(f"B4 (Bottom-Left): {b4}")
        
        if selected_image_idx in scale_factors:
            scale_factor = scale_factors[selected_image_idx]
            b1_nm = (b1[0] * scale_factor, b1[1] * scale_factor)
            b2_nm = (b2[0] * scale_factor, b2[1] * scale_factor)
            b3_nm = (b3[0] * scale_factor, b3[1] * scale_factor)
            b4_nm = (b4[0] * scale_factor, b4[1] * scale_factor)
            st.write("Fixed Corner Coordinates (nm):")
            st.write(f"B1 (Top-Left): ({b1_nm[0]:.2f}, {b1_nm[1]:.2f})")
            st.write(f"B2 (Top-Right): ({b2_nm[0]:.2f}, {b2_nm[1]:.2f})")
            st.write(f"B3 (Bottom-Right): ({b3_nm[0]:.2f}, {b3_nm[1]:.2f})")
            st.write(f"B4 (Bottom-Left): ({b4_nm[0]:.2f}, {b4_nm[1]:.2f})")
        
        # Relative width and height
        relative_width_pixels = b2[0] - b1[0]
        relative_height_pixels = b4[1] - b1[1]
        st.write(f"Relative Width (B2 - B1 X): {relative_width_pixels} pixels")
        st.write(f"Relative Height (B4 - B1 Y): {relative_height_pixels} pixels")
        if selected_image_idx in scale_factors:
            relative_width_nm = relative_width_pixels * scale_factor
            relative_height_nm = relative_height_pixels * scale_factor
            st.write(f"Relative Width (nm): {relative_width_nm:.2f}")
            st.write(f"Relative Height (nm): {relative_height_nm:.2f}")
        
        # Sliders for two arbitrary points P1 and P2 to measure distance
        st.write("Select two points P1 and P2 to measure distance:")
        col1, col2 = st.columns(2)
        with col1:
            p1_x = st.slider("P1 X Position", 0, width - 1, 0)
            p1_y = st.slider("P1 Y Position (from top)", 0, height - 1, 0)
        with col2:
            p2_x = st.slider("P2 X Position", 0, width - 1, width // 2)
            p2_y = st.slider("P2 Y Position (from top)", 0, height - 1, height // 2)
        
        # Draw markers and line on the image
        img_with_markers = img.copy()
        cv2.circle(img_with_markers, (p1_x, p1_y), 5, (255, 0, 0), -1)  # P1
        cv2.circle(img_with_markers, (p2_x, p2_y), 5, (0, 255, 0), -1)  # P2
        cv2.line(img_with_markers, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)  # Line between P1 and P2
        
        st.image(img_with_markers, caption=f"Image {selected_image_idx+1} with Points P1 (red), P2 (green), and Distance Line (blue)", use_column_width=True)
        
        # Compute distance between P1 and P2
        distance_pixels = np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
        st.write(f"Distance between P1 ({p1_x}, {p1_y}) and P2 ({p2_x}, {p2_y}) (pixels): {distance_pixels:.2f}")
        
        if selected_image_idx in scale_factors:
            distance_nm = distance_pixels * scale_factor
            st.write(f"Distance between P1 and P2 (nm): {distance_nm:.2f}")

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
            s1_x, s1_y = scale_bar_coords[idx][0]
            s2_x, s2_y = scale_bar_coords[idx][1]
            height, width = img.shape[:2]

            # Calculate corner coordinates in nm (top-left as 0,0)
            top_left_nm = (0, 0)
            top_right_nm = (width * scale_factor, 0)
            bottom_right_nm = (width * scale_factor, height * scale_factor)
            bottom_left_nm = (0, height * scale_factor)

            st.write(f"Coordinate System (nm) for Image {idx+1}:")
            st.write(f"Top-Left: ({top_left_nm[0]:.2f}, {top_left_nm[1]:.2f})")
            st.write(f"Top-Right: ({top_right_nm[0]:.2f}, {top_right_nm[1]:.2f})")
            st.write(f"Bottom-Right: ({bottom_right_nm[0]:.2f}, {bottom_right_nm[1]:.2f})")
            st.write(f"Bottom-Left: ({bottom_left_nm[0]:.2f}, {bottom_left_nm[1]:.2f})")

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
