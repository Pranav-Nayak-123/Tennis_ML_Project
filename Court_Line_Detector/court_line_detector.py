import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        # Load ResNet-50 architecture with a custom final layer
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 28)  # 14 pairs (x, y) coordinates
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        # Image transformation pipeline for input compatibility
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Convert image to RGB for consistency
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        # Get keypoint predictions
        with torch.no_grad():
            outputs = self.model(image_tensor).squeeze()

        keypoints = outputs.cpu().numpy()

        # Normalize points to fit the frame dimensions
        h, w = img_rgb.shape[:2]
        keypoints[::2] = np.clip(keypoints[::2] * w / 224, 0, w)  # X coordinates
        keypoints[1::2] = np.clip(keypoints[1::2] * h / 224, 0, h)  # Y coordinates

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Draw keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            # Ensure keypoints are within bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(image, f"{i // 2}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        # Overlay keypoints on each video frame
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
