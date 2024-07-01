import os
import cv2
import pandas as pd
import numpy as np

def extract_features(image):
    r, c = image.shape
    N = r * c
    
    # Mean
    mean = np.mean(image)
    
    # Variance
    variance = np.var(image)
    
    # Standard Deviation
    sd = np.std(image)
    
    # Mean Absolute Deviation
    mad = np.mean(np.abs(image - mean))
    
    # Median Absolute Deviation
    median = np.median(image)
    medad = np.median(np.abs(image - median))
    
    # Local Contrast
    lc = np.max(image) - np.min(image)
    
    # Local Percentage for gray scale value 175
    lp = np.sum(image == 175) / N
    
    # Percentile 25
    percentile_25 = np.percentile(image, 25)
    
    # Percentile 75
    percentile_75 = np.percentile(image, 75)
    
    # Custom GLCM feature calculation
    def calculate_glcm_features(image):
        glcm = np.zeros((256, 256), dtype=np.float32)
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                glcm[image[i, j], image[i, j + 1]] += 1

        glcm /= glcm.sum()
        
        contrast = 0
        dissimilarity = 0
        homogeneity = 0
        ASM = 0
        correlation = 0
        mean_i = np.mean(image)
        mean_j = np.mean(image)
        std_i = np.std(image)
        std_j = np.std(image)
        
        for i in range(256):
            for j in range(256):
                contrast += (i - j) ** 2 * glcm[i, j]
                dissimilarity += np.abs(i - j) * glcm[i, j]
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
                ASM += glcm[i, j] ** 2
                correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)

        energy = np.sqrt(ASM)
        
        return contrast, dissimilarity, homogeneity, ASM, energy, correlation

    contrast, dissimilarity, homogeneity, ASM, energy, correlation = calculate_glcm_features(image)
    
    features = [
        mean, variance, sd, mad, medad, lc, lp,
        percentile_25, percentile_75,
        contrast, dissimilarity, homogeneity, ASM, energy, correlation
    ]
    return features

def create_csv(dataset_path, output_csv):
    # Create an empty DataFrame with column names
    columns = [
        'image_id', 'mean', 'variance', 'sd', 'mad', 'medad', 'lc', 'lp',
        'percentile_25', 'percentile_75',
        'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation',
        'class'
    ]
    df = pd.DataFrame(columns=columns)

    # Iterate over each class folder in the dataset
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                # Load the image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Extract features from the image
                features = extract_features(image)
                # Append the image ID and class label to the features
                features = [image_name] + features + [class_name]
                # Append the features to the DataFrame
                df.loc[len(df)] = features

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

# Define the paths to the dataset and the output CSV file
train_dataset_path = 'train'
test_dataset_path = 'test'
train_output_csv = 'train_features.csv'
test_output_csv = 'test_features.csv'

# Create CSV files for the training and testing datasets
create_csv(train_dataset_path, train_output_csv)
create_csv(test_dataset_path, test_output_csv)
