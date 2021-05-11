# Bag-Detection-and-counting
Object detection is a process of finding all the possible instances of real-world objects, such as human faces, flowers, cars, etc. in images or videos, in real-time with utmost accuracy. The object detection technique uses derived features and learning algorithms to recognize all the occurrences of an object category. The real-world applications of object detection are image retrieval, security and surveillance, advanced driver assistance systems, also known as ADAS, and many others

How Does Object Detection Technique Work :-
              Deep learning uses a multi-layer approach to extract high-level features from the data that is provided to it. It doesn’t require the features to be provided manually for classification, instead, it tries to transform its data into an abstract representation. It simply learns by examples and uses it for future classification. Deep learning is influenced by the artificial neural networks (ANN) present in our brains.
              Most of the deep learning methods implement neural networks to achieve the results. All the deep learning models require huge computation powers and large volumes of labeled data to learn the features directly from the data. The day to day applications of deep learning is news aggregation or fraud news detection, visual recognition, natural language processing, etc.
              
              
1.	Taking the visual as an input, either by an image or a video.
2.	Divide the input visual into sections, or regions.
3.	Take each section individually, and work on it as a single image
4.	Passing these images into our Convolutional Neural Network (CNN) to classify them into possible classes.
5.	After the classification, we can combine all the images and generate the original input image, but also with the detected objects and their labels.
 
Counting:-
          One simple but often ignored use of object detection is counting. The ability to count people, cars, flowers, and even microorganisms, is a real world need that is broadly required for different types of systems using images. Recently with the ongoing surge of video surveillance devices, there’s a bigger than ever opportunity to turn that raw information into structured data using computer vision
Single Shot Detector (Object Detection):-
            SSD (Single Shot Detector) is reviewed. By using SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as Rcnn series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.
            SSD300 achieves 74.3% mAP at 59 FPS while SSD500 achieves 76.9% mAP at 22 FPS

 
    

  

 

