# Basketball Analysis on Landmarks with LSTMs (B.A.L.L)

Bagherlee, Kian | Davila-Bugarin, Rafael | Holden, Matthew

### Abstract

Free throws are crucial for winning basketball games. To identify factors that affect players' ability to score them, several studies have been conducted, particularly in the computer vision field. We contribute to this research by introducing Basketball Analysis on Landmarks with LSTMs (B.A.L.L), a methodology that combines Google's Mediapipe Pose Landmark Model, YOLO v8 by Ultralytics, and a custom trained LSTM model for predicting a binary “make” or “miss” classification. We addressed the “garbage in, garbage out” problem by increasing and improving the dataset's quality. As a result, we achieved a maximum accuracy of 65% and a maximum F1 score of 67%.

### Data

Gathering data for this project provided us with a valuable learning experience. Initially, we followed Purnama et al. (2024) and recorded ourselves shooting 100 times at a video quality of 30 frames per second (FPS). However, the ball's speed made it difficult to recognize it as an object in the frames, often appearing more like a football than a basketball (see Appendix I). To address this issue, we conducted a second session, recording over 200 videos at 60 FPS with six participants. This resulted in a dataset of 116 successful shots and 120 misses, ensuring a balanced dataset for model training. Both sessions were recorded from the Free Throw line at approximately a 30 degree angle and 6 feet to the right of the shooter. The view from the camera ensured a balanced view of both body landmarks and basketball trajectory after release.  We had 5 males and 1 female participants with an average scoring rate of 49%.

### Methodology

![image](https://github.com/user-attachments/assets/8c0b77bf-1fdf-45b0-a033-d93dad39fa0c)


First we converted each video into frames, then we used Mediapipe Pose Landmark by Google to generate 33 points of the body for each frame. Mediapipe is a bundle that uses a MobileNetV2-like (Sandler et al, 2019) CNN and the Generative Human Shape pipeline to estimate 3D body poses in real-time. That gave us 33 body parts as we can see in the figure 1.a. 

![image](https://github.com/user-attachments/assets/7f936f53-be4d-42c0-a62a-a341eb55484e)

To normalize the frames and ensure consistency, we picked a baseline video as a reference and reoriented the coordinate system of all other videos to this baseline video. Reorientation to the baseline video consists of applying an X and Y coordinate displacement to the body landmark locations and basketball location to ensure the hips of the shooter always align at the same starting position for every video, even if the camera angle varies slightly. This standardization helps make the data more reliable for the model. 

To predict success or failure among participants, we employed a model architecture consisting of two Long Short-Term Memory (LSTM) blocks (Hochreiter et al., 1997) with a sequence length of 30 timesteps each. Videos were preprocessed by sampling frames to standardize their length to 30 frames per video, producing time-series keypoint data as input to the LSTM model.

The first layer of the model is a unidirectional LSTM. This LSTM has an equal number of input and hidden neurons (32), configured to identify and highlight key temporal patterns within the input time-series data. The output of this LSTM, consisting of the processed sequence, is then passed to a Bidirectional LSTM. This bidirectional structure was selected to capture both forward and backward dependencies in the time-series data, providing a more comprehensive analysis of temporal relationships.

To prevent overfitting, a dropout layer with a probability of 0.10 was applied to the outputs of the last LSTM layer. This placement helps regularize the model by randomly dropping connections during training. Both LSTM layers consist of a single layer, and the Bidirectional LSTM has 32 hidden neurons in total (16 neurons per direction).

The fully connected layer at the output of the Bidirectional LSTM maps the extracted features to the final output layer, which produces logits. These logits are then passed to a CrossEntropy loss function for classification. The Adam optimizer was used to minimize the loss during training, with a learning rate of 0.001. A batch size of 16 was selected to balance computational efficiency with model performance.

This model architecture, including the choice of standardized input length, regularization strategies, and hyperparameter tuning, was designed to address the challenges posed by having a small dataset and to maximize the model's ability to generalize.

### Results and Discussion

The first B.A.L.L. implementation (Baseline) used the low quality dataset, around hundred videos with 30 FPS, only pose-estimation data, no feature engineering and no ball detection, as the ball could not be detected (to see how the ball looks with 30 FPS and 60 FPS go to Appendix I). The second implementation was built with a larger and higher quality dataset (more than two hundred videos at 60FPS) while still only using pose-estimation as input to the LSTM. The third model included the 60 FPS data, pose estimation, and ball detection. Our final model added biomechanical feature engineering. The table below shows the results for each model.

![image](https://github.com/user-attachments/assets/40b9c7a0-a456-4ce4-b1b9-61397f91d19c)

![image](https://github.com/user-attachments/assets/f9b2e623-aaab-4f7f-ba9e-e7b3b355fd04)

The results from the experiments display a few interesting behaviors. First, the baseline model with the 30 FPS videos has better-than-random accuracy, but a 0% F1-Score. To get a F-1 Score this low, taking into consideration the accuracy, it was determined that the model was predicting a miss for every single entry. For a combination of reasons surrounding the low dataset size, poor data quality, and lack of extra features, this model performed well below expectations. When the quality of the dataset was improved (from 30FPS to 60FPS), the higher frames per second resulted in less motion blur and enabled more accurate body landmark estimations from the MediaPipe model. This improved input data to the LSTM model along with the larger dataset size (100 videos to >200 videos) is reflected in the improved accuracy and F1 scores of the second model. For the third model with basketball detection, some of the training data was unable to be processed due to inaccurate basketball locations. As a result, our overall dataset size was reduced and we suspect this affected the model results. Even with the reduced dataset, we still see an increase in accuracy indicating that the basketball location tracking was value-added information for model prediction. However, with such a limited dataset to begin with, removing a fraction of the training data proved impactful to overall performance. Despite this slightly reduced dataset, the final model with biomechanical feature engineering (joint angle calculations, head-ball planar angle calculation, etc.) showed significantly improved F1 score and results similar to the Human-Pose Only model. While only nine additional biomechanical features were calculated for each frame, these additional features proved valuable to the model and counteracted the negative effects of the reduced dataset size. Overall, we see that higher quality data, more data, and the introduction of basketball tracking and critical biomechanical features improved the model’s predictive capability.

### Conclusion

B.A.L.L has successfully demonstrated that getting landmarks for the body and basketball, utilizing them to extract key biomechanical features, and iterating over an entire video to mimic time series data, all provide a LSTM model enough information to learn how to identify a successfully made free throw. The landmarks on the body give a great starting baseline, and adding more features, whether another landmark for the ball or certain critical body angles, help improve the reasoning ability of the model in total. Future work in this regard would be spent on more feature engineering, identifying joint angles not tested prior. This was all seen through a very limited dataset, which B.A.L.L. was also able to demonstrate being equally as important. Most importantly, the number of available examples is incredibly vital, and any future work would include expanding the available dataset. B.A.L.L. also demonstrated how the quality of these are important, and having an even distribution (scores/fail, male/female, shooting form, etc.) helps with the generalization of the model. Future work would also be spent on the model architecture, and different ways to improve. The biggest addition next would be the implementation of an attention mechanism, allowing the model to better focus on the landmarks that are most crucial. Achieving such success with the detailed limiting factors showcases the effectiveness of B.A.L.L., and how effective LSTMs truly are in determining the success of free throws through tracking landmarks.

