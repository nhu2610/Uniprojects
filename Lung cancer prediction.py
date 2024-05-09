import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
import os


# Define the output folder path
output_folder = r"E:\BIP\Final project\processed_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def apply_Butterworth_LPF(img, D0=20, n=2):
    # Convert image to float32 and grayscale
    imgFloat32 = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # The height and width of the picture
    rows, cols = imgFloat32.shape

    # Centralized 2D array f (x, y) * - 1 ^ (x + y)
    mask = np.ones(imgFloat32.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    fImage = imgFloat32 * mask
    
    # Reduce image size if necessary
    scaledRows = cv2.getOptimalDFTSize(rows)
    scaledCols = cv2.getOptimalDFTSize(cols)
    
    # Fast Fourier transform
    dftImage = np.zeros((scaledRows, scaledCols), np.float32)
    dftImage[:rows, :cols] = fImage
    dftImage = cv2.dft(dftImage, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_REAL_OUTPUT)
    dftAmp = cv2.magnitude(dftImage[:, :, 0], dftImage[:, :, 1])
    dftAmpLog = np.log(1.0 + dftAmp)
    dftAmpNorm = np.uint8(cv2.normalize(dftAmpLog, None, 0, 255, cv2.NORM_MINMAX))
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(dftAmp)
    
    # Construction of low pass filter transfer function
    u, v = np.mgrid[0:scaledRows:1, 0:scaledCols:1]
    D = np.sqrt(np.power((u - maxLoc[1]), 2) + np.power((v - maxLoc[0]), 2))
    epsilon = 1e-8  # Prevent division by 0
    lpFilter = 1.0 / (1.0 + np.power(D / (D0 + epsilon), 2 * n))
    
    # Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
    dftLPfilter = dftImage * lpFilter[:, :, np.newaxis]
    
    # The inverse Fourier transform is performed on the low-pass Fourier transform, and only the real part is taken
    idft = cv2.idft(dftLPfilter, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # Centralized 2D array g (x, y) * - 1 ^ (x + y)
    mask2 = np.ones(dftAmp.shape)
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    idftCen = idft * mask2
    
    # Truncation function, limiting the value to [0, 255]
    result = np.clip(idftCen, 0, 255)
    imgBLPF = result.astype(np.uint8)
    imgBLPF = imgBLPF[:rows, :cols]
    return imgBLPF
def augment_image(image):
    # Define a sequence of image augmentation operations
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20)),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 2))
    ])
    
    # Apply the sequence of augmentation operations to the input image
    augmented_image = seq.augment_image(image)
    
    return augmented_image
#Normalization image
def normalize_image(img):
    # Convert image to grayscale if it is not already
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE normalization to the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_normalized = clahe.apply(img)
    return img_normalized

# import all image files with the .jpg extension
images = glob.glob (r"E:\BIP\Final project/*.jpg") #read images from folder

image_data = []
for img in images:
    this_image = cv2.imread(img, 1)
    this_image1 = cv2.resize(this_image, (128, 128))  # resize image to a common size
    this_image2 =  apply_Butterworth_LPF(this_image1)
    this_image3 = normalize_image(this_image2)
    this_image4 = augment_image(this_image3)
    # Plot histogram for raw image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(this_image.flatten(), bins=256, color='b', alpha=0.7)
    plt.title('Raw Image Histogram')

# Plot histogram for normalized image
    plt.subplot(1, 2, 2)
    plt.hist(this_image3.flatten(), bins=256, color='r', alpha=0.7)
    plt.title('Normalized Image Histogram')

# Show the plots
    plt.tight_layout()
    plt.show()

    _, filename = os.path.split(img)

    # Construct the output path for the processed image
    output_path = os.path.join(output_folder, filename)
    # Save the processed image to the output folder
    cv2.imwrite(output_path, this_image)
    image_data.append(this_image4)
    # Define the output folder path for intermediate steps
    output_folder = r"E:\BIP\Final project\ImageTest"
    os.makedirs(output_folder, exist_ok=True)

# Save the intermediate steps of each image
    for i, image in enumerate(image_data):
        output_path = os.path.join(output_folder, f"Image_Test_{i}.jpg")
        cv2.imwrite(output_path, image)

