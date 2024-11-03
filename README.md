# Snow Detection Algorithm with Synthetic Data and Ensemble Models

This repository contains the codebase for the research presented in three key papers, including the thesis titled **"Developing a Snow Detection Algorithm Using Spatial Attention for Pedestrian Safety"**. This project focuses on detecting snow on sidewalks to enhance pedestrian safety, particularly aiding vulnerable populations like the elderly and visually impaired. The code integrates synthetic data augmentation, ensemble modeling, and advanced convolutional neural network (CNN) architectures.

## Project Overview

The goal of this project is to create a reliable snow detection model using real-world and synthetic image datasets. The model leverages spatial attention mechanisms, synthetic image generation via diffusion models, and an ensemble of VGG-19 and ResNet-50 CNN architectures. This combined approach enhances the accuracy and robustness of snow detection on sidewalks under varied conditions, supporting real-time detection for mobile device users.

### Key Features

- **Spatial Attention Mechanisms**: Enhances feature detection in challenging snow environments.
- **Synthetic Image Augmentation**: Utilizes generated images to compensate for limited real-world snow data.
- **Ensemble Modeling**: Combines VGG-19 and ResNet-50 predictions for improved performance.
- **Real-Time Implementation**: Designed for deployment on mobile devices to alert pedestrians of snowy conditions.

## Usage

1. **Prepare Input Data**: Ensure images are in the correct format.
2. **Run Detection**: Execute the main script to classify sidewalks as snowy or clear.
3. **Review Results**: Output classifications and metrics are saved for analysis.

## Evaluation

The model’s performance was evaluated using both real-world and synthetic datasets, with metrics like F1 score, accuracy, and recall. Ensemble methods and individual model metrics were compared, highlighting the potential and limitations of synthetic data for training robust snow detection models.

## Research Papers

This project draws from the following studies:

- **Developing a Snow Detection Algorithm Using Spatial Attention for Pedestrian Safety**: Details the core spatial attention model and its application for pedestrian safety.
- **Snow Classification Using Prompt-Based Generated Images**: Investigates the effectiveness of synthetic images in training snow detection models, highlighting differences in model performance between synthetic and real-world datasets.
- **<a href='https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1014&context=mwais2024'>Image Classification for Snow Detection to Improve Pedestrian Safety</a>**: Presents the ensemble model approach using VGG-19 and ResNet-50 CNNs, achieving accuracy and F1 scores of over 70% on a custom dataset of snowy sidewalks.

## License

This project is licensed under the Apache-2.0 License. See the <a href='https://github.com/Neatherblok/SnowDetection/blob/main/LICENSE'>LICENSE</a> file for details.

## Acknowledgments

Special thanks to Dr. Rajeev Bukralia and Dr. Mansi Bhavsar for their guidance and to Minnesota State University, Mankato, for supporting this research.
