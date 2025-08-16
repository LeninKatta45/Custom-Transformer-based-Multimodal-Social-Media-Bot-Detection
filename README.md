
# Social Media Bot Detection using Multimodal Transformer

## Overview

**Social Media Bot Detection** is a cutting-edge project aimed at identifying automated bot accounts on social media platforms using a multimodal transformer approach. This system analyzes images, textual posts, and metadata to predict whether an account is a bot, enhancing the integrity of social media interactions. The solution integrates advanced deep learning techniques and is deployable in cloud environments.

## Key Features

- **Multimodal Transformer:** Leverages images, textual data, and metadata for robust bot detection with a combined vision and traditional transformer architectures.
- **BERT Embeddings:** Utilizes BERT for effective textual embeddings, and Autoencoder to reduce the dimentionaly of the feature maps improving the model's understanding of language nuances and images features.
- **Scalable Architecture:** Designed for cloud deployment, ensuring high availability and performance.
- **CI/CD Integration:** Continuous integration and deployment with GitHub Actions for streamlined updates.

## Performance Metrics

- **Accuracy:** 90.20%
- **Precision:** 75%
- **Recall:** 92.31%
- **F1 Score:** 82.76%
- **AUC-ROC:** 92.51%

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Frameworks:** PyTorch, Transformers (Hugging Face)
- **Web Frameworks:** Flask, FastAPI
- **Cloud Platform:** AWS
- **CI/CD:** GitHub Actions for automated integration and deployment

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/LeninKatta45/SocialMediaBotDetection.git
    cd SocialMediaBotDetection
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Launch the application:

    ```bash
    python app.py
    ```

2. Access the application at `http://localhost:5000` to analyze social media accounts and detect bot activity.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the GNU General Public License (GPL). See the `LICENSE` file for more details.

## Contact

For any queries or feedback, please reach out:

- **Lenin Balaji Katta**
- [Email](mailto:b21ai020@kitsw.ac.in)
- [LinkedIn](https://www.linkedin.com/in/leninkatta)
- [GitHub](https://github.com/LeninKatta45)
