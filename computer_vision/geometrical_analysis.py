import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit app title
st.title("Core-Shell Nanoparticle Analysis (Black-and-White TEM)")

# File uploader for TEM images
uploaded_files = st.file_uploader("Upload TEM Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Check if at least one image is uploaded
if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        images.append(np.array(image))
    
    # Display uploaded images
    st.subheader("Uploaded Images")
    for i, img in enumerate(images):
        st.image(img, caption=f"Image {i+1}", use_column_width=True)

    # Scale bar input
    st.subheader("Scale Bar Calibration")
    scale_bar_nm = st.number_input("Enter scale bar length (nm):", min_value=1.0, value=100.0)
    scale_bar_pixels = st.number_input("Enter scale bar length in pixels:", min_value=1, value=100)
    scale_factor = scale_bar_nm / scale_bar_pixels  # nm/pixel

    # Segmentation parameters
    st.subheader("Segmentation Parameters")
    core_threshold = st.slider("Core Intensity Threshold (0-255)", 0, 255, (0, 100))
    shell_threshold = st.slider("Shell Intensity Threshold (0-255)", 0, 255, (100, 200))
    min_particle_area = st.number_input("Minimum Particle Area (pixels):", min_value=10, value=100)

    # Process images button
    if st.button("Analyze Images"):
        all_core_radii = []
        all_shell_thicknesses = []

        for idx, img in enumerate(images):
            st.write(f"Processing Image {idx+1}...")

            # Apply thresholding for core and shell
            core_mask = cv2.inRange(img, core_threshold[0], core_threshold[1])
            shell_mask = cv2.inRange(img, shell_threshold[0], shell_threshold[1])

            # Morphological operations to clean masks
            kernel = np.ones((3, 3), np.uint8)
            core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
            shell_mask = cv2.morphologyEx(shell_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            core_contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shell_contours, _ = cv2.findContours(shell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and pair contours
            core_radii = []
            shell_thicknesses = []
            img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            for core_cnt in core_contours:
                if cv2.contourArea(core_cnt) > min_particle_area:
                    # Fit circle to core
                    (x, y), core_radius_pixels = cv2.minEnclosingCircle(core_cnt)
                    core_radius_nm = core_radius_pixels * scale_factor
                    core_radii.append(core_radius_nm)

                    # Find corresponding shell contour (closest enclosing contour)
                    for shell_cnt in shell_contours:
                        if cv2.contourArea(shell_cnt) > min_particle_area:
                            (x_shell, y_shell), shell_radius_pixels = cv2.minEnclosingCircle(shell_cnt)
                            # Check if shell encloses core (based on center proximity and size)
                            if abs(x - x_shell) < 10 and abs(y - y_shell) < 10 and shell_radius_pixels > core_radius_pixels:
                                shell_thickness_nm = (shell_radius_pixels - core_radius_pixels) * scale_factor
                                shell_thicknesses.append(shell_thickness_nm)
                                # Draw contours
                                cv2.drawContours(img_with_contours, [core_cnt], -1, (255, 0, 0), 1)  # Core in blue
                                cv2.drawContours(img_with_contours, [shell_cnt], -1, (0, 255, 0), 1)  # Shell in green
                                break

            all_core_radii.extend(core_radii)
            all_shell_thicknesses.extend(shell_thicknesses)

            # Display image with contours
            st.image(img_with_contours, caption=f"Image {idx+1}: Core (Blue), Shell (Green)", use_column_width=True)

        # Calculate and display averages
        avg_core_radius = np.mean(all_core_radii) if all_core_radii else 0
        avg_shell_thickness = np.mean(all_shell_thicknesses) if all_shell_thicknesses else 0
        st.write(f"Average Core Radius: {avg_core_radius:.2f} nm")
        st.write(f"Average Shell Thickness: {avg_shell_thickness:.2f} nm")

        # Plot histogram of core radii
        if all_core_radii:
            fig, ax = plt.subplots()
            ax.hist(all_core_radii, bins=20, color="blue", alpha=0.7)
            ax.set_xlabel("Core Radius (nm)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Plot histogram of shell thicknesses
        if all_shell_thicknesses:
            fig, ax = plt.subplots()
            ax.hist(all_shell_thicknesses, bins=20, color="green", alpha=0.7)
            ax.set_xlabel("Shell Thickness (nm)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
