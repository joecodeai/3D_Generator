from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64

# Define the improved UNet model architecture with additional layers
class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        self.down1 = self.conv_block(1, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 1024)
        self.down6 = self.conv_block(1024, 2048)  # Additional downsampling layer
        
        self.up1 = self.upconv_block(2048, 1024)  # Adjust input channels to match additional layer
        self.up2 = self.upconv_block(2048, 512)
        self.up3 = self.upconv_block(1024, 256)
        self.up4 = self.upconv_block(512, 128)
        self.up5 = self.upconv_block(256, 64)
        self.final_depth = nn.Conv2d(128, 1, kernel_size=1)  # Depth map output
        self.final_rgb = nn.Conv2d(128, 3, kernel_size=1)    # RGB image output
        
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for RGB output
    
    def conv_block(self, in_channels, out_channels, use_dropout=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # Additional convolution layer
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_channels, out_channels, use_dropout=True):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # Additional convolution layer
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        d4 = self.down4(F.max_pool2d(d3, 2))
        d5 = self.down5(F.max_pool2d(d4, 2))
        d6 = self.down6(F.max_pool2d(d5, 2))  # Additional downsampling step
        
        up1 = self.up1(d6)
        up1 = torch.cat((up1, d5), dim=1)
        up2 = self.up2(up1)
        up2 = torch.cat((up2, d4), dim=1)
        up3 = self.up3(up2)
        up3 = torch.cat((up3, d3), dim=1)
        up4 = self.up4(up3)
        up4 = torch.cat((up4, d2), dim=1)
        up5 = self.up5(up4)
        up5 = torch.cat((up5, d1), dim=1)
        
        depth_output = self.final_depth(up5)
        rgb_output = self.sigmoid(self.final_rgb(up5))  # Apply Sigmoid activation
        
        return depth_output, rgb_output


app = Flask(__name__)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the trained model and move it to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedUNet().to(device)
model.load_state_dict(torch.load('improved_model_weights.pth', map_location=device))
model.eval()

def depth_to_point_cloud(depth_map, rgb_image, scale=1000.0, focal_length=500.0):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map * scale

    x = (x - width / 2.0) * z / focal_length
    y = (y - height / 2.0) * z / focal_length

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

    return points, colors

def image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['file']
    image = Image.open(file).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_output, rgb_output = model(image)

    depth_output = depth_output.squeeze(0).cpu().numpy()
    rgb_output = rgb_output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Normalize depth to range [0, 1]
    depth_output = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min())
    # Scale depth for better visualization
    depth_output = depth_output * 10

    # Normalize RGB to range [0, 255] and convert to uint8
    rgb_output = (rgb_output * 255).astype(np.uint8)

    points, colors = depth_to_point_cloud(depth_output[0], rgb_output)

    # Convert images to base64
    depth_image_base64 = image_to_base64((depth_output[0] * 255).astype(np.uint8))
    rgb_image_base64 = image_to_base64(rgb_output)

    # Convert points and colors to JSON-serializable format
    points_list = points.tolist()
    colors_list = colors.tolist()

    return jsonify({
        "depth_image": depth_image_base64,
        "rgb_image": rgb_image_base64,
        "points": points_list,
        "colors": colors_list
    })

if __name__ == '__main__':
    app.run(debug=True)