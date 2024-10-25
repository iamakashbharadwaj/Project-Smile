import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageColor
import torch
import torchvision
from io import BytesIO
from streamlit_cropper import st_cropper

# Function to create a radial gradient
def create_radial_gradient(size, center_color, edge_color):
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center
            dist = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
            max_dist = np.sqrt((width // 2) ** 2 + (height // 2) ** 2)
            # Calculate the interpolation factor
            alpha = dist / max_dist
            gradient[y, x, 0] = int(center_color[0] * (1 - alpha) + edge_color[0] * alpha)  # R
            gradient[y, x, 1] = int(center_color[1] * (1 - alpha) + edge_color[1] * alpha)  # G
            gradient[y, x, 2] = int(center_color[2] * (1 - alpha) + edge_color[2] * alpha)  # B

    return gradient

# Function to convert PIL image to bytes
def pil_to_bytes(img, format='PNG'):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr.read()

st.title("LinkedIn Display Picture Editor")
st.write("I made this application because I was tired of logging into websites to remove backgrounds and having to pay for downloading HD images.")
st.write("It's optimised for viewing angles in LinkedIn for that was my sole purpose of frustration. If you do not like the auto smart background gradient you can simply download it as a transparent png file and add your own ")
st.write(" You can improve the code base by sending requests to the repo")
st.write(" It works well enough for my purposes, please donot send any requests asking for my time to add new features")
st.write(" You are free to fork and use this code as your own and expand upon it. Good Luck - Akash Bharadwaj")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Cropping functionality
    st.write("Crop the image:")
    cropped_img = st_cropper(img, aspect_ratio=(1, 1))

    # Preprocess the cropped image for background removal
    st.write("Removing background...")

    # Load pre-trained DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(cropped_img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # Get the mask for person (class 15)
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = output_predictions == 15  # 15 corresponds to the 'person' class

    # Convert mask to binary format (0 and 1)
    mask = np.uint8(mask)

    # Find the largest connected component (assuming it's the person)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background label 0

    # Create a refined mask containing only the largest component
    refined_mask = np.zeros_like(mask)
    refined_mask[labels == largest_label] = 1

    # Apply morphological operations to clean up the mask
    kernel = np.ones((10, 10), np.uint8)
    refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)  # Larger dilation
    refined_mask = cv2.erode(refined_mask, kernel, iterations=1)

    # Apply Gaussian blur to smooth the edges of the mask
    refined_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (5, 5), sigmaX=1)

    # Threshold to create a binary mask again after blurring
    refined_mask = (refined_mask > 0.5).astype(np.uint8)

    # Apply the mask to the original image
    img_np = np.array(cropped_img)
    mask_3d = np.stack([refined_mask] * 3, axis=-1)  # Create 3-channel mask for RGB

    # Create a new image with an alpha channel
    img_with_alpha = np.dstack((img_np, np.where(mask_3d[:, :, 0], 255, 0)))  # 255 for foreground, 0 for background

    # Convert to PIL Image and display
    img_with_alpha_pil = Image.fromarray(img_with_alpha.astype('uint8'), 'RGBA')

    # Color selection for gradient
    outer_color = img_np[refined_mask > 0].mean(axis=0).astype(int)  # Average color of the outer image
    outer_color_rgb = tuple(outer_color)
    edge_color = tuple(255 - np.array(outer_color_rgb))  # Complementary color

    # Layout for the features
    col1, col2, col3 = st.columns(3)

    # Background removal column
    with col1:
        st.subheader("Remove Background")
        st.image(img_with_alpha_pil, caption='Background Removed', use_column_width=True)
        # Convert image to bytes for download
        img_bytes = pil_to_bytes(img_with_alpha_pil)
        st.download_button("Download Image", img_bytes, "removed_background.png", "image/png")

    # Upscaling column
    with col2:
        st.subheader("Upscale Image")
        # AI Upscaling Logic
        if st.button("Upscale Image"):
            # Upscaling Logic Here
            img_with_alpha_tensor = torchvision.transforms.ToTensor()(img_with_alpha_pil).unsqueeze(0)
            # For now, just simulating the upscaling with a placeholder
            upscaled_img = img_with_alpha_pil.resize((img_with_alpha_pil.size[0] * 2, img_with_alpha_pil.size[1] * 2), Image.LANCZOS)

            st.image(upscaled_img, caption='Upscaled Image', use_column_width=True)
            # Convert upscaled image to bytes for download
            upscaled_img_bytes = pil_to_bytes(upscaled_img)
            st.download_button("Download Upscaled Image", upscaled_img_bytes, "upscaled_image.png", "image/png")

    # Gradient application column
    with col3:
        st.subheader("Apply Gradient")
        # Button to apply gradient
        if st.button("Apply Gradient"):
            gradient = create_radial_gradient(img_with_alpha_pil.size, outer_color_rgb, edge_color)

            # Create an image from the gradient
            gradient_img = Image.fromarray(gradient)

            # Composite the cropped image with the gradient background
            final_image = Image.alpha_composite(gradient_img.convert('RGBA'), img_with_alpha_pil)

            # Display the final image
            st.image(final_image, caption='Final Image with Gradient Background', use_column_width=True)
            # Convert final image to bytes for download
            final_image_bytes = pil_to_bytes(final_image)
            st.download_button("Download Image with Gradient", final_image_bytes, "image_with_gradient.png", "image/png")
