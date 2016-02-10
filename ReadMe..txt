---------
1. Canny
---------
Error rate: Desired edge detection filter should find all the edges, there should not be any missing edges,
and it should respond only to edge regions.
Localization: Distance between detected edges and actual edges should be as small as possible.
Response: The edge detector should not identify multiple edge pixels where only a single edge exists.

The program implements Canny edge detection for the given input image, we will first smooth the images, then compute gradients, magnitude, and orientation of the gradient. This procedure is followed by non-max suppression, and finally hysteresis thresholding is applied to finalize the steps.

Berkeley Segmentation Dataset Training images have been used for this program.

-------------
2. Evaluation
-------------
The program reports quantitative evaluation of the edge detection results implemented in the with the above program with the following functions:

The following metrics have been used to evaluate edge detector with respect to the edge maps provided by Berkeley along with input images.

Sensitivity (or true positive rate TPR) as TP=(TP + FN)
Specificity (or true negative rate TNR) as TN=(TN + FP)
Precision (or positive predictive value PPV) as TP=(TP + FP)
Negative Predictive Value as TN=(TN + FN)
Fall-out (or false positive rate FPR) as FP=(FP + TN)
False Negative Rate FNR as FN=(FN + TP)
False Discovery Rate FDR as FP=(FP + TP)
Accuracy as (TP + TN)=(TP + FN + TN + FP)
F-score as 2TP=(2TP + FP + FN)
Matthew’s Correlation Coefficient as TP*TN-FP*FN/sqrt[((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))]
-----------------------------------------------------------------------------------------
We then add Gaussian and Salt-Pepper noise to the input image and repeat the evaluation step.

---------
3.ENTROPY
---------
The program computes the summation of entropy A and entropy B, then total entropy=H(A)+
H(B). Note that A and B correspond to background and foreground of the input image.

The Gray scale images and corresponding binary images for the input image are plotted. (binary images
are obtained through thresholding gray-scale images at threshold level T. H(T) = H(A) + H(B).)


NO ADDITIONAL IMAGE PROCESSING TOOLS WERE USED TO PROGRAM!!!