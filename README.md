# Real-Time Sign Language Alphabet Recognition  

This is a real-time sign language alphabet recognition program that uses Google MediaPipe to extract hand landmark data. A Multi-Layer Perceptron (MLP) model then processes this data to recognize sign language letters.  

## **Demo**  
Here is a demonstration of the program recognizing the Filipino Sign Language (FSL) alphabet:  
![Demo](demo.gif)

Watch the full demo here:
https://github.com/manganeseheptoxide/Sign-Language-Alphabet-Recognition/assets/demo.mkv  

## **Datasets**  
The model is trained using the following datasets:  
- **American Sign Language (ASL) Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/chaitanyakakade77/american-sign-language-dataset)  
- **Filipino Sign Language (FSL) Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/japorton/fsl-dataset)  

## **Landmark Data Processing**  
To convert image datasets into landmark data, I created a separate program. You can find it here:  
➡️ [Image-to-Landmark Data Processor](https://github.com/manganeseheptoxide/Image-To-Landmark-Data-Processor)  
