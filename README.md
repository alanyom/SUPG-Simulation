Intro
I used the Amazon Review dataset from the Stanford Network Analysis Project. 
My goal was to explore sentiment classification using the 4th algorithm from the SUPG research paper on budget proxy labeling. 
I tested this algorithm’s importance sampling approach against traditional uniform sampling.

Task 1
Improvements / Long-Term Considerations: 
Scalability: For larger datasets, I could optimize the process of sentiment analysis and sampling by implementing batch processing or distributing the tasks. 
Learning: An online learning approach where the model updates as new reviews come in would improving the model’s predictions over time. 
Model Customization: I could fine-tuning a transformer model for sentiment analysis specifically tailored to Amazon reviews.


Dataset source
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv


Importance Sampling
Uniform Sampling
Trial #1:
Best Tau: 0.47
F1 Score: 0.8831
Precision: 0.9615
Recall: 0.8166

Upper Bound: 1.6428
Lower Bound: -0.6428

Accuracy: 0.7834
Confusion Matrix: [[35464  4628]
 [12703 27205]]
Trial #1:
Best Tau: 0.16
F1 Score: 0.8296
Precision: 0.7568
Recall: 0.9180

Upper Bound: 0.9205
Lower Bound: 0.0795

Accuracy: 0.7260
Confusion Matrix: [[20704 19388]
 [ 2530 37378]]
Trial #2:
Best Tau: 0.32
F1 Score: 0.8440
Precision: 0.8407
Recall: 0.8474

Upper Bound: 1.5651
Lower Bound: -0.5651

Accuracy: 0.7923
Confusion Matrix: [[31050  9036]
 [ 7582 32332]]
Trial #2: 
Best Tau: 0.21
F1 Score: 0.8319
Precision: 0.7705
Recall: 0.9038

Upper Bound: 0.9067
Lower Bound: 0.0933

Accuracy: 0.7615
Confusion Matrix: [[24999 15087]
 [ 3995 35919]]
Trial #3:
Best Tau: 0.17
F1 Score: 0.8645
Precision: 0.8030
Recall: 0.9361

Upper Bound: 1.3927
Lower Bound: -0.3927

Accuracy: 0.7349
Confusion Matrix: [[21729 18347]
 [ 2861 37063]]
Trial #3:
Best Tau: 0.28
F1 Score: 0.8211
Precision: 0.7647
Recall: 0.8864

Upper Bound: 0.8897
Lower Bound: 0.1103

Accuracy: 0.7859
Confusion Matrix: [[29341 10735]
 [ 6396 33528]]
Trial #4:
Best Tau: 0.17
F1 Score: 0.8325
Precision: 0.7309
Recall: 0.9669

Upper Bound: 1.3313
Lower Bound: -0.3313

Accuracy: 0.7358
Confusion Matrix: [[21695 18303]
 [ 2836 37166]]
Trial #4: 
Best Tau: 0.53
F1 Score: 0.8537
Precision: 0.9211
Recall: 0.7955

Upper Bound: 0.7997
Lower Bound: 0.2003

Accuracy: 0.7691
Confusion Matrix: [[36585  3413]
 [15060 24942]]
Trial #5:
Best Tau: 0.34
F1 Score: 0.8431
Precision: 0.8548
Recall: 0.8317

Upper Bound: 1.5689
Lower Bound: -0.5689

Accuracy: 0.7924
Confusion Matrix: [[31883  8219]
 [ 8392 31506]]
Trial #5:
Best Tau: 0.20
F1 Score: 0.8333
Precision: 0.7937
Recall: 0.8772

Upper Bound: 0.8802
Lower Bound: 0.1198

Accuracy: 0.7576
Confusion Matrix: [[24438 15664]
 [ 3730 36168]]
Summary
Mean accuracy: 0.76776
Mean precision: 0.83818
Mean recall: 0.87974
Summary
Mean accuracy: 0.76002
Mean precision: 0.80928
Mean recall: 0.87618




Conclusion:

While importance sampling showed slightly better performance across some metrics, the improvement was not significant. 
This suggests that in the Amazon Reviews dataset, where most samples have similar information value, importance sampling may not offer substantial benefits. 
In contrast, the SUPG paper’s use cases, like the hummingbird CNN, involve highly skewed or rare classes, where importance sampling can prioritize the most valuable data for labeling and model training.


