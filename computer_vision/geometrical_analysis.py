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
    scale_bar_nms = {}  # Store nm per image

    for i, img in enumerate(images):
        st.write(f"Initial Calibration for Image {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            s1_x = st.number_input(f"S1 X for Image {i+1}", min_value=0, max_value=img.shape[1]-1, value=0, key=f"s1_x_init_{i}")
            s1_y = st.number_input(f"S1 Y for Image {i+1}", min_value=0, max_value=img.shape[0]-1, value=0, key=f"s1_y_init_{i}")
        with col2:
            s2_x = st.number_input(f"S2 X for Image {i+1}", min_value=0, max_value=img.shape[1]-1, value=img.shape[1]//2, key=f"s2_x_init_{i}")
            s2_y = st.number_input(f"S2 Y for Image {i+1}", min_value=0, max_value=img.shape[0]-1, value=img.shape[0]//2, key=f"s2_y_init_{i}")
        
        scale_bar_nm = st.number_input(f"Scale bar length (nm) for Image {i+1}", min_value=1.0, value=20.0, key=f"nm_init_{i}")
        
        if st.button(f"Confirm Initial Scale Bar for Image {i+1}"):
            scale_bar_coords[i] = [(s1_x, s1_y), (s2_x, s2_y)]
            scale_bar_nms[i] = scale_bar_nm
            pixel_distance = np.sqrt((s2_x - s1_x)**2 + (s2_y - s1_y)**2)
            if pixel_distance > 0:
                scale_factors[i] = scale_bar_nm / pixel_distance
                st.success(f"Initial Scale factor for Image {i+1}: {scale_factors[i]:.2f} nm/pixel")
            else:
                st.error("Invalid scale bar points: zero distance detected.")

    # Interactive Measurement Tool
    st.subheader("Interactive Measurement Tool")
    selected_image_idx = st.selectbox("Select Image for Measurement", options=range(len(images)), format_func=lambda x: f"Image {x+1}")
    if selected_image_idx is not None:
        img = images[selected_image_idx]
        height, width = img.shape[:2]
        
        # Static corner points B1-B4 (top-left as (0,0))
        b1 = (0, 0)  # Top-left
        b2 = (width - 1, 0)  # Top-right
        b3 = (width - 1, height - 1)  # Bottom-right
        b4 = (0, height - 1)  # Bottom-left
        
        st.write("Static Corner Coordinates B_i (pixels):")
        st.write(f"B1 (Top-Left): {b1}")
        st.write(f"B2 (Top-Right): {b2}")
        st.write(f"B3 (Bottom-Right): {b3}")
        st.write(f"B4 (Bottom-Left): {b4}")
        
        # Dynamic corner points D1-D4, initial to B
        st.write("Adjust Dynamic Corner Points D_i:")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            d1_x = st.slider("D1 X", 0, width - 1, b1[0], key=f"d1_x_{selected_image_idx}")
            d1_y = st.slider("D1 Y", 0, height - 1, b1[1], key=f"d1_y_{selected_image_idx}")
            d2_x = st.slider("D2 X", 0, width - 1, b2[0], key=f"d2_x_{selected_image_idx}")
            d2_y = st.slider("D2 Y", 0, height - 1, b2[1], key=f"d2_y_{selected_image_idx}")
        with col_d2:
            d3_x = st.slider("D3 X", 0, width - 1, b3[0], key=f"d3_x_{selected_image_idx}")
            d3_y = st.slider("D3 Y", 0, height - 1, b3[1], key=f"d3_y_{selected_image_idx}")
            d4_x = st.slider("D4 X", 0, width - 1, b4[0], key=f"d4_x_{selected_image_idx}")
            d4_y = st.slider("D4 Y", 0, height - 1, b4[1], key=f"d4_y_{selected_image_idx}")
        
        d1 = (d1_x, d1_y)
        d2 = (d2_x, d2_y)
        d3 = (d3_x, d3_y)
        d4 = (d4_x, d4_y)
        
        # Relative width and height based on D_i (assuming rectangular alignment)
        relative_width_pixels = abs(d2[0] - d1[0])
        relative_height_pixels = abs(d4[1] - d1[1])
        st.write(f"Relative Width (D2 X - D1 X): {relative_width_pixels} pixels")
        st.write(f"Relative Height (D4 Y - D1 Y): {relative_height_pixels} pixels")
        
        if selected_image_idx in scale_factors:
            scale_factor = scale_factors[selected_image_idx]
            relative_width_nm = relative_width_pixels * scale_factor
            relative_height_nm = relative_height_pixels * scale_factor
            st.write(f"Relative Width (nm): {relative_width_nm:.2f}")
            st.write(f"Relative Height (nm): {relative_height_nm:.2f}")
        
        # Adjust S1 S2 with sliders
        st.write("Adjust Scale Bar Points S1 and S2:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            s1_x = st.slider("S1 X", 0, width - 1, scale_bar_coords.get(selected_image_idx, [(0,0)])[0][0] if selected_image_idx in scale_bar_coords else 0)
            s1_y = st.slider("S1 Y", 0, height - 1, scale_bar_coords.get(selected_image_idx, [(0,0)])[0][1] if selected_image_idx in scale_bar_coords else 0)
        with col_s2:
            s2_x = st.slider("S2 X", 0, width - 1, scale_bar_coords.get(selected_image_idx, [(0,0), (width//2, height//2)])[1][0] if selected_image_idx in scale_bar_coords else width//2)
            s2_y = st.slider("S2 Y", 0, height - 1, scale_bar_coords.get(selected_image_idx, [(0,0), (width//2, height//2)])[1][1] if selected_image_idx in scale_bar_coords else height//2)
        
        scale_bar_nm = st.number_input("Scale bar length (nm)", min_value=1.0, value=scale_bar_nms.get(selected_image_idx, 20.0))
        
        if st.button("Recalculate Scale Factor"):
            pixel_distance = np.sqrt((s2_x - s1_x)**2 + (s2_y - s1_y)**2)
            if pixel_distance > 0:
                scale_factors[selected_image_idx] = scale_bar_nm / pixel_distance
                scale_bar_coords[selected_image_idx] = [(s1_x, s1_y), (s2_x, s2_y)]
                scale_bar_nms[selected_image_idx] = scale_bar_nm
                st.success(f"Updated Scale factor: {scale_factors[selected_image_idx]:.2f} nm/pixel")
            else:
                st.error("Invalid scale bar points: zero distance detected.")
        
        s1 = (s1_x, s1_y)
        s2 = (s2_x, s2_y)
        
        # Points P1-P4 for circle approximation
        st.write("Adjust Points P1-P4 for Circle Approximation:")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p1_x = st.slider("P1 X", 0, width - 1, width // 4)
            p1_y = st.slider("P1 Y", 0, height - 1, height // 4)
            p2_x = st.slider("P2 X", 0, width - 1, 3 * width // 4)
            p2_y = st.slider("P2 Y", 0, height - 1, height // 4)
        with col_p2:
            p3_x = st.slider("P3 X", 0, width - 1, 3 * width // 4)
            p3_y = st.slider("P3 Y", 0, height - 1, 3 * height // 4)
            p4_x = st.slider("P4 X", 0, width - 1, width // 4)
            p4_y = st.slider("P4 Y", 0, height - 1, 3 * height // 4)
        
        p1 = (p1_x, p1_y)
        p2 = (p2_x, p2_y)
        p3 = (p3_x, p3_y)
        p4 = (p4_x, p4_y)
        
        # Fit circle to P1-P4
        def fit_circle(points):
            if len(points) < 3:
                return None, None, None
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            A = np.c_[2*x, 2*y, np.ones(len(points))]
            b = x**2 + y**2
            try:
                c = np.linalg.lstsq(A, b, rcond=None)[0]
                xc = c[0]
                yc = c[1]
                r = np.sqrt(xc**2 + yc**2 + c[2])
                return xc, yc, r
            except:
                return None, None, None
        
        center_x, center_y, radius = fit_circle([p1, p2, p3, p4])
        if center_x is not None:
            st.write(f"Circle Center (pixels): ({center_x:.2f}, {center_y:.2f})")
            st.write(f"Approximate Radius (pixels): {radius:.2f}")
            if selected_image_idx in scale_factors:
                scale_factor = scale_factors[selected_image_idx]
                center_x_nm = center_x * scale_factor
                center_y_nm = center_y * scale_factor
                radius_nm = radius * scale_factor
                st.write(f"Circle Center (nm): ({center_x_nm:.2f}, {center_y_nm:.2f})")
                st.write(f"Approximate Radius (nm): {radius_nm:.2f}")
        else:
            st.warning("Could not fit circle to points.")
        
        # Draw all points and labels on the image
        img_with_labels = img.copy()
        
        # Function to draw point and label
        def draw_label(img, point, label, color=(255, 255, 255), offset=(10, 10)):
            cv2.circle(img, (int(point[0]), int(point[1])), 3, color, -1)
            text = f"{label} ({int(point[0])}, {int(point[1])})"
            cv2.putText(img, text, (int(point[0]) + offset[0], int(point[1]) + offset[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Colors for different groups
        color_b = (255, 0, 0)  # Blue for B
        color_d = (0, 255, 0)  # Green for D
        color_s = (0, 0, 255)  # Red for S
        color_p = (255, 255, 0)  # Cyan for P
        
        # Draw B_i
        draw_label(img_with_labels, b1, "B1", color_b)
        draw_label(img_with_labels, b2, "B2", color_b, offset=(-50, 10))  # Adjust offset to avoid going out
        draw_label(img_with_labels, b3, "B3", color_b, offset=(-50, -10))
        draw_label(img_with_labels, b4, "B4", color_b, offset=(10, -10))
        
        # Draw D_i
        draw_label(img_with_labels, d1, "D1", color_d)
        draw_label(img_with_labels, d2, "D2", color_d, offset=(-50, 10))
        draw_label(img_with_labels, d3, "D3", color_d, offset=(-50, -10))
        draw_label(img_with_labels, d4, "D4", color_d, offset=(10, -10))
        
        # Draw S_i if available
        if selected_image_idx in scale_bar_coords:
            draw_label(img_with_labels, s1, "S1", color_s)
            draw_label(img_with_labels, s2, "S2", color_s)
            cv2.line(img_with_labels, s1, s2, color_s, 1)
        
        # Draw P_i
        draw_label(img_with_labels, p1, "P1", color_p)
        draw_label(img_with_labels, p2, "P2", color_p)
        draw_label(img_with_labels, p3, "P3", color_p)
        draw_label(img_with_labels, p4, "P4", color_p)
        
        # Draw approximated circle if fitted
        if center_x is not None:
            cv2.circle(img_with_labels, (int(center_x), int(center_y)), int(radius), (255, 0, 255), 1)  # Magenta circle
            draw_label(img_with_labels, (center_x, center_y), "Center", (255, 0, 255), offset=(10, 20))
        
        st.image(img_with_labels, caption=f"Image {selected_image_idx+1} with Labeled Points", use_column_width=True)

    # Segmentation parameters and analysis remain the same
    st.subheader("Segmentation Parameters")
    red_threshold = st.slider("Red Cu Core Threshold (0-255)", 0, 255, (150, 255), key="red")
    green_threshold = st.slider("Green Ag Shell Threshold (0-255)", 0, 255, (50, 150), key="green")
    min_particle_area = st.number_input("Minimum Particle Area (pixels):", min_value=10, value=100)

    if st.button("Analyze Images"):
        for idx, img in enumerate(images):
            if idx not in scale_factors or scale_factors[idx] is None:
                st.write(f"Skipping Image {idx+1}: Scale bar not calibrated.")
                continue

            st.write(f"Processing Image {idx+1}...")
            scale_factor = scale_factors[idx]
            height, width = img.shape[:2]

            # Coordinate system in nm (top-left as 0,0)
            top_left_nm = (0, 0)
            top_right_nm = (width * scale_factor, 0)
            bottom_right_nm = (width * scale_factor, height * scale_factor)
            bottom_left_nm = (0, height * scale_factor)

            st.write(f"Coordinate System (nm) for Image {idx+1}:")
            st.write(f"Top-Left: ({top_left_nm[0]:.2f}, {top_left_nm[1]:.2f})")
            st.write(f"Top-Right: ({top_right_nm[0]:.2f}, {top_right_nm[1]:.2f})")
            st.write(f"Bottom-Right: ({bottom_right_nm[0]:.2f}, {bottom_right_nm[1]:.2f})")
            st.write(f"Bottom-Left: ({bottom_left_nm[0]:.2f}, {bottom_left_nm[1]:.2f})")

            # Image processing for segmentation
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            lower_red = np.array([0, 100, red_threshold[0]])
            upper_red = np.array([10, 255, red_threshold[1]])
            lower_green = np.array([40, 50, green_threshold[0]])
            upper_green = np.array([80, 255, green_threshold[1]])

            red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                                cv2.drawContours(img_with_contours, [red_cnt], -1, (0, 0, 255), 1)
                                cv2.drawContours(img_with_contours, [green_cnt], -1, (0, 255, 0), 1)
                                break

            st.image(img_with_contours, caption=f"Image {idx+1}: Red Cu Core, Green Ag Shell", use_column_width=True)

            avg_core_radius = np.mean(core_radii) if core_radii else 0
            avg_shell_radius = np.mean(shell_radii) if shell_radii else 0
            avg_shell_thickness = np.mean(shell_thicknesses) if shell_thicknesses else 0
            st.write(f"Average Cu Core Radius: {avg_core_radius:.2f} nm")
            st.write(f"Average Ag Shell Radius: {avg_shell_radius:.2f} nm")
            st.write(f"Average Shell Thickness: {avg_shell_thickness:.2f} nm")

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
