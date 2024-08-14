# mallSegmentaionKMeanClustering
<h1>Mall Customer Segmentation (K-Means Clustering)</h1>

<h2>Purpose</h2>
To determine the optimal number of clusters for K-Means on Mall Customer Data.
<br />


<h2>Languages </h2>

- <b>Python</b> 
  

<h2>Environments Used </h2>

- <b>Windows 10</b> (21H2)

<h2>Data Sources and Dataset Description </h2>

This study focuses on a dataset of about 22,000 student essays from a massive open online course (MOOC). The essays were written in response to a standardized prompt, asking students to apply theoretical knowledge to practical problems. This provides a valuable source of natural language data.
 <br/>


<h2>Dataset Composition and Access </h2>

- <b> Document Identifier: Each essay is uniquely identified, allowing for precise referencing and analysis.</b>
- <b> Full Text: Essays are presented in their entirety in UTF-8 format, ensuring that the data remains consistent and accessible across different systems. </b>
- <b> Tokens and Annotations: Utilizing the SpaCy English tokenizer, essays are broken down into tokens. Annotations follow the BIO (Beginning, Inner, Outer) format, which helps in identifying and classifying different types 
    of PII. For example, tokens labeled as “B-NAME_STUDENT” signify the beginning of a student’s name, while “I-NAME_STUDENT” indicates a continuation. </b>

<h2>Types of PII to Be Identified </h2>
For the purposes of this project, we focused on detecting seven specific types of PII within the essays, each representing a unique privacy concern:


- <b> NAME_STUDENT: Names directly associated with individual students.</b>
- <b> EMAIL: Email addresses that could be used to contact or identify a student.</b>
- <b> USERNAME: Usernames that might be linked to student profiles on various platforms.</b>
- <b> ID_NUM: Identifiable numbers, such as student IDs or social security numbers.</b>
- <b> PHONE_NUM: Telephone numbers associated with students.</b>
- <b> URL_PERSONAL: Personal URLs that could directly or indirectly identify a student.</b>
- <b> STREET_ADDRESS: Residential addresses, either complete or partial, that are tied to students.</b>

<h2>Calculating and Analyzing Label Frequencies </h2>
Understanding the distribution of labels within dataset was crucial. Therefore, by calculating the frequency of each label, we gained insights into the prevalence of different types of PII, which helped in prioritizing certain types for more focused analysis and model training

<p align="center">
<img src="https://imgur.com/RnMMPNT.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2>Label Mapping and Replacement </h2>
To facilitate easier processing and improve readability during model training, mapping numerical labels to their corresponding string representations using a predefined dictionary was necessary. This step was essential for maintaining clarity and consistency across the dataset. For instance, numerical labels were converted as follows:

- <b> B-NAME_STUDENT.</b>
- <b> I-NAME_STUDENT.</b>

<h2>Data Filtering and Token Validation</h2>
Filtered rows of the DataFrame based on specific label values to focus on significant or underrepresented PII types. Additionally, we implemented a function to validate tokens, ensuring that only relevant and correctly formatted data was retained for model training. This step helped eliminate corrupt or outlier data that could skew the model’s learning.


<h2>Organizing Data at the Document Level</h2>
This step was crucial for preserving the narrative and contextual integrity of each essay. We then iterated over tokens and labels for each document, allowing for a detailed examination to ensure that the data accurately followed our labeling schema.

<h2>Preparing Data for Modeling</h2>
Finally, the tokens were concatenated into sentences, and the label sequences were converted into lists of strings. This preparation was crucial because it transformed the tokenized data into a format suitable for sequential processing by our chosen NLP models. This step ensures that when models like BERT and ELECTRA process the data, it is in an optimal state for efficient and effective processing.

<h2>Exploring Different Models: XLNet, LSTM, and GRU in PII Detection</h2>
As part of our comprehensive approach to detecting personally identifiable information (PII) in educational datasets, various machine learning models were experimented with, each offering unique strengths and challenges. While the primary focus was on the performance of BERT and ELECTRA, the capabilities of XLNet, LSTM, and GRU were also explored, yielding mixed results in the context of PII detection.

<h2>XLNet: Advanced Permutation-Based Modeling</h2>
XLNet builds on the Transformer-XL model using Permutation Language Modeling. This method enables XLNet to understand bidirectional contexts by evaluating all possible permutations of the input tokens. This theoretically allows for a more comprehensive grasp of the text’s context. However, in our PII detection task, XLNet did not perform as well as BERT and ELECTRA. This performance gap might be attributed to XLNet’s complexity and its general training approach, which may not align perfectly with the specific nuances and varieties of PII within educational texts.

<h2>LSTM and GRU: The Challenges with Recurrent Networks</h2>
LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units) are types of recurrent neural networks (RNNs) renowned for their effectiveness in sequence prediction tasks across various NLP applications. These models process data sequentially, which is beneficial for capturing temporal dependencies in text data.

However, in our project, both LSTM and GRU faced difficulties in managing the complex dependencies and contextual nuances required for effective PII identification. Their sequential nature, while powerful for many tasks, proved less effective than Transformer-based models in handling the intricacies of detecting and classifying dispersed and nuanced PII data in large texts. The challenges were particularly pronounced when dealing with the scattered nature of PII in student essays, where context plays a crucial role in accurately identifying sensitive information.


<h2>Choosing Electra for Enhanced PII Detection (Architecture and Implementation):</h2>
ELECTRA, an acronym for Efficiently Learning an Encoder that Classifies Token Replacements Accurately, marks a significant departure from traditional language model training approaches like Masked Language Modeling (MLM) used by BERT. ELECTRA introduces the Replaced Token Detection (RTD) technique, which innovatively utilizes every token in the input by substituting some tokens with plausible yet incorrect alternatives. This method allows the model to assess the authenticity of each token, enabling comprehensive learning from the entire input sequence, thereby maximizing data efficiency and training effectiveness.


 <b> Advantages of ELECTRA’s RTD Method : </b>
 
Unlike traditional MLM that only learns from approximately 15% of the data (the masked tokens), ELECTRA’s RTD approach evaluates every token in the input, thus learning more effectively from the full context. This leads to faster learning rates and reduces the computational resources required, allowing ELECTRA to achieve or even surpass benchmarks set by other advanced models like RoBERTa and XLNet with significantly less computational overhead.
 
 <b> ELECTRA’s Dual Model Architecture : </b>
 
ELECTRA uses a dual model architecture with two transformer models: a generator and a discriminator. This setup is similar to generative adversarial networks (GANs) but is designed for language processing without the adversarial aspect. The generator creates realistic token replacements, while the discriminator checks if each token in the sequence is real or a replacement. This interaction improves the model's ability to understand and process complex language patterns, enhancing its generalization capabilities for various NLP tasks.


<h2>Project Implementation and PII Detection with ELECTRA:</h2>
This project aimed at detecting personally identifiable information (PII) from large datasets by utilizing ELECTRA's advanced NLP capabilities. The process started with a thorough data preparation phase, leveraging the pandas library to organize and preprocess our data efficiently. This involved managing missing values and logically grouping tokens by document, laying a solid groundwork for the modeling process.

To address the challenge of processing large text volumes, we used the SlidingWindowDataset class. This method enabled us to divide the text into segments of 512 tokens with a stride of 128, crucial for preserving context in lengthy documents. This approach was both memory efficient and vital for ensuring the accuracy of PII classification across extensive datasets.

<h2>Performance Insights from the Classification Report:</h2>
Delving into the classification report, the model’s performance varied across different classes:

- <b>The most common class, denoted as ‘0’, achieved near-perfect precision and recall, indicating the model’s proficiency in correctly identifying non-PII tokens.</b>

- <b>The precision and recall for classes such as ‘1’ (0.85 precision, 0.31 recall) and ‘2’ (0.88 precision, 0.61 recall) show a considerable rate of accurate predictions but also suggest room for improvement in recall,     
   particularly in retrieving all relevant instances of these PII types.</b>
   
- <b>Classes ‘3’, ‘5’, ‘7’, ‘8’ depict challenges in detection, with both precision and recall at zero, highlighting the difficulty the model faced with rare PII types, possibly due to insufficient training examples.</b>

- <b>Other classes like ‘9’, ‘10’, and ‘11’ demonstrated low recall, indicating that while the model could identify these PII types to some degree, it often missed them in the dataset.</b>

<p align="center"> <img src="https://i.imgur.com/GwYkEZ6.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />

<h2>Reflecting on the Model’s Effectiveness:</h2>

The model achieved an impressive overall accuracy of 99% in detecting PII, highlighting its effectiveness across the dataset. However, disparities were observed in the macro and weighted averages, particularly in detecting rarer types of PII, indicating a need for further development in this area.

<h2>Mastering PII Detection with BERT: Architecture and Mechanism</h2>

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a groundbreaking language representation model developed by Google. It fundamentally changes the way machines understand human language by using a mechanism known as bidirectional training. This model is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, BERT is pre-trained on a large corpus of text and then fine-tuned for specific tasks, which allows it to achieve state-of-the-art results in a wide range of language processing tasks.

<h2>The Transformer in BERT </h2>

At the core of BERT lies the Transformer, a sophisticated deep learning model employing self-attention mechanisms. Unlike earlier models that processed words sequentially, the Transformer processes each word in relation to all others in the sentence. This architecture, first introduced by Vaswani et al. in 'Attention is All You Need,' empowers BERT to grasp contextual nuances between words bidirectionally. This marks a substantial advancement over previous models that treated sentences as unidirectional sequences of words.

<h2>BERT’s Encoder Architecture </h2>

The BERT model uses what is known as the Transformer encoder architecture. Each encoder consists of two primary components:

- <b>Self-Attention Mechanism:</b> Allows BERT to consider the context of each word in the sentence to determine its meaning, rather than considering words in isolation.

- <b>Feed-Forward Neural Networks:</b> Each layer also contains a small feed-forward neural network that processes each word position separately. The output of these networks is what the next layer processes if there are subsequent layers.

<h2>Processes in BERT’s Encoder</h2>

- <b> Tokenization:</b>BERT begins processing its input by breaking down text into tokens.
- <b> Input Embedding:</b> Each token is then converted into numerical vectors that represent various linguistic features of the token.
- <b> Positional Encoding:</b> BERT adds positional encodings to the input embeddings to express the position of each word within the sentence.
- <b> Self-Attention:</b> The self-attention layers in the encoders allow each token to interact with all other tokens in the input layer, focusing more on the relevant tokens.
- <b> Layer Stacking: </b>Multiple layers of the Transformer encoder allow BERT to learn rich, context-dependent representations of the input text.
- <b> Output:</b> The final output is a series of vectors, one for each input token, which are then used for specific NLP tasks.

The BERT model’s ability to process words in relation to all other words in a sentence provides a deeper sense of language context and nuance, making it exceptionally effective for tasks requiring a deep understanding of language structure and context, such as PII detection.

<h2>BERT’s Implementation in PII Detection: Methodology, Results, and Impact</h2>

<b> Finetuning a Bert Model: </b> At the heart of our efforts was the BertForTokenClassification model, fine-tuned to navigate our dataset’s imbalances. To better detect less frequent PII types, we incorporated a weighted loss function, giving due importance to rarer labels.

- <b> Sliding Window Tokenization:</b> As shown in the code chunk below, his technique addressed the challenges of lengthy texts, preserving the entirety of information for the model to consider.
<p align="center"> <img src="https://i.imgur.com/anQgzlV.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />

<h2>Handle Class Imbalance</h2>
<p align="center"> <img src="https://i.imgur.com/YBhNp4I.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />
"To tackle class imbalance in classification tasks, we compute weights that are inversely proportional to the frequency of each class within the dataset. These weights undergo normalization to ensure that the most prevalent class receives a weight of one, maintaining a standardized scale. Subsequently, we define a CrossEntropyLoss function incorporating these customized weights. We set -101 for the ignore_index parameter, allowing the model to disregard specific tokens (such as padding in sequence models) during loss computation. This approach effectively mitigates the impact of class imbalance by prioritizing the learning of less frequent classes."

<h2>Customized Training and Validation Cycles:</h2>
<p align="center"> <img src="https://i.imgur.com/VbVKHdM.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />
<p align="center"> <img src="https://i.imgur.com/VY6faoq.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />
  
The above code chunk aims at token classification tasks, using PyTorch and the Hugging Face Transformers library. The process begins by setting a seed for reproducibility across different runs, ensuring consistent initialization and behavior in stochastic operations.

- <b> Imports and Setup:</b> Necessary libraries are imported, including PyTorch for tensor operations and deep learning, Hugging Face’s Transformers for accessing pre-trained models and utilities, and Scikit-learn for evaluation metrics. Dataloader classes from PyTorch are used to handle data batching and shuffling.

- <b> Batch Accuracy Calculation: </b> A helper function batch_accuracy calculates the accuracy of predictions in each batch. It filters out certain tokens (with a label of -101, typically used for padding or non-evaluated tokens) before comparing the predicted labels with true labels to compute accuracy.

- <b>Seed Initialization:</b> The randomness is controlled by setting seeds for libraries like NumPy and PyTorch, ensuring that the results are reproducible.
  
- <b> Learning Rate Retrieval:</b> A function get_current_learning_rate is defined to fetch the current learning rate from the optimizer, useful for monitoring and adjustments during training.
  
- <b> Training Loop Setup:</b> The training loop is initialized with conditions to track the best validation loss and to implement early stopping if there’s no improvement in validation loss for a defined number of epochs (patience).

- <b>Training and Validation Phases:</b> During each epoch, the model undergoes training and validation phases. In the training phase, the model parameters are updated using the gradient descent method based on the loss computed from the model’s output and the actual labels. The validation phase evaluates the model on a separate dataset to monitor performance and avoid overfitting.

- <b>Loss and Accuracy Tracking:</b> For both training and validation, loss and accuracy are calculated. This information is used to determine model performance and make adjustments if necessary.
  
- <b>Model Saving and Early Stopping:</b> The best-performing model parameters are saved, and training can be halted early if there’s no improvement in validation loss over several epochs, preventing unnecessary computations and potential overfitting.

<h2>Understanding the Classification Report</h2>

The classification report provides a quantitative insight into the model’s performance:

<p align="center"> <img src="https://i.imgur.com/u9UGGXu.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />

- <b> Test Loss and Accuracy:</b> The test loss stood at 0.0413, indicating the model’s prediction error, while the test accuracy reached an impressive 95.07%, reflecting the model’s ability to correctly classify the tokens as PII or non-PII.

<h1>Performance by Category:</h1>

- <b> High Precision Classes: </b> Classes like ‘B-EMAIL’ and ‘I-PHONE_NUM’ showed high precision, indicating that when the model predicted these classes, it was correct most of the time.
- <b> High Recall Classes: </b> The ‘B-NAME_STUDENT’ and ‘I-STREET_ADDRESS’ had high recall, meaning the model was able to identify most instances of these classes within the data.
- <b> Challenges with Rare PII Types:</b> Categories like ‘B-ID_NUM’ and ‘I-URL_PERSONAL’ had precision and recall scores of 0, suggesting the model struggled to identify these rare instances. This could be due to the small number of examples in the training data.

  <h1>F1-Score and Support:</h1>

- The F1-score, a harmonic mean of precision and recall, was notably high for general classes but lower for rare types, indicating uneven model performance across different PII categories.</b>
- The ‘support’ column, indicating the number of true instances for each label, helps to explain the disparity in F1-scores — categories with low support struggled with lower F1-scores.
  
  <h1>Results and Impact</h1>
  
By integrating BERT into our PII detection initiative, we significantly improved our capability to identify and protect sensitive information within educational texts. The model’s high accuracy and efficient processing of voluminous texts demonstrate the transformative potential of advanced NLP technologies in safeguarding privacy in the digital educational landscape.


<h2>Conclusion and Future Directions for PII Detection in Education </h2>
- <b>Reflecting on Our Journey:</b>

The exploration into PII detection in educational datasets has been both challenging and enlightening. By applying sophisticated NLP models like BERT and ELECTRA, we’ve made significant strides in protecting student privacy. Our methodology, from advanced tokenization to the nuanced application of BERT’s fine-tuned capabilities, has allowed us to detect a wide range of PII with high accuracy and efficiency.

- <b>Achievements and Learnings:</b>
It is observed that while BERT excels in recognizing and classifying common PII types, it encounters hurdles with rarer PII categories. The classification report illuminated these disparities, guiding our steps towards model improvement. Our commitment to ensuring comprehensive protection of sensitive information while maintaining the utility of educational datasets for research has been at the forefront of our efforts.

  <h1>Charting the Path Forward</h1>

The field of PII detection is ever-evolving, and our work here is just the beginning. Future directions include:

- <b>Enhanced Sampling and Representation: </b> We plan to augment our dataset with more examples of rare PII to balance the representation across categories.
- <b>Algorithmic Refinement: </b> We aim to adjust model weights and explore alternative algorithms that might offer better performance, particularly in the context of rare PII detection.
- <b>Continuous Learning and Adaptation: </b> Keeping pace with the evolving nature of language and data privacy concerns, we’ll ensure our models continue learning from new data, enhancing their accuracy and reliability.

<h2>CONCLUSION</h2>

<b> As the  PII detection models continue to be refine, commitments to safeguarding student privacy remains unwavering. The journey ahead is promising, and dedication to advancing these tools to ensure educational data serves its purpose ethically and responsibly.</b>
