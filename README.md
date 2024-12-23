# DL_LAB
# DATA255 - Fall 2024 - Labs 1 & 2

## Lab 1: Deep Learning and Object Detection

### Part 1: Deep Learning-Based Recommendation System (10 Points)
- Implement a **Wide and Deep Learning** recommender system for the Anime Dataset.
- Use 80/20 train-test split and record the prediction accuracy.
- Dataset links:
  - [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)

### Part 2: Image Classification with Deep Learning (40 Points)
1. **Dataset**: [Sports Image Classification](https://www.kaggle.com/datasets/sidharkal/sports-image-classification/data)
2. Tasks:
   - Explain fundamental concepts like gradient descent, regularization, and activation functions.
   - Visualize and summarize dataset characteristics.
   - Train and optimize a neural network, experimenting with layers, epochs, activation functions, loss functions, and optimizers.
   - Evaluate performance using metrics like loss vs. epochs and F1 score vs. epochs.

### Part 3: Object Detection (50 Points)
- Dataset: [Road Sign Dataset](https://drive.google.com/drive/folders/1LIceJIn69vzmn40eAqz5kYJrsrC6m5lc?usp=sharing)
- Tasks:
  - Perform data augmentation and transformations.
  - Implement an object detection model from scratch and compare with pre-trained models like YOLOv8.
  - Evaluate using IOU and document observations with hyperparameters.

---

## Lab 2: Advanced GANs, NLP, and RAG Models

### Part 1: Exploring GAN Latent Space with StyleGAN (25 Points)
- Implement **StyleGAN** using PyTorch.
- Train the model using the [provided dataset](https://arxiv.org/pdf/1812.04948).

### Part 2: NLP Competition (25 Points)
- Task: Classify comments into categories such as toxicity, obscene, threat, etc., using non-transformer-based models.
- Dataset: Available via a [Kaggle competition](https://www.kaggle.com/t/a69bfc3629fe4e1cb3c2ade298b76e9e).
- Submission: Kaggle prediction file and markdown comments in a `.ipynb` file.

### Part 3: Retrieval-Augmented Generation (RAG) with LangChain (30 Points)
- Build a **RAG model** to process PDF documents and enable user queries.
- Utilize models like ChatGPT, Llama2, or Mistral alongside FAISS or ChromaDB.
- Evaluate using ROUGE scores and document quality metrics.

---

## Submission Guidelines
For each lab, submit:
1. **Code notebooks** with clear documentation.
2. **Reports** covering:
   - Model architecture, challenges, and experiments.
   - Observations and hyperparameter tuning.
3. Relevant files (e.g., weights, predicted images).
