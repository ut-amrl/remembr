import cv2
import os
import argparse

def get_images_from_directory(directory):
    # Get all image files from directory and sort them
    images = [img for img in os.listdir(directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Sort files to maintain order
    return images

def convert_images_to_video(image_dir, output_video, fps=30):
    images = get_images_from_directory(image_dir)
    if not images:
        print("No images found in the directory.")
        return
    
    # Read first image to get dimensions
    first_image_path = os.path.join(image_dir, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi, 'mp4v' for .mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        resized_frame = cv2.resize(frame, (width, height))  # Ensure all frames are the same size
        video_writer.write(resized_frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in a directory to a video.")
    parser.add_argument("--image_dir", help="Directory containing images")
    parser.add_argument("--output_video", help="Output video file (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second (default: 30)")
    
    args = parser.parse_args()
    
    convert_images_to_video(args.image_dir, args.output_video, args.fps)
