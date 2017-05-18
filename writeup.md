## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[notcar]: ./output_images/notcar.png
[car_YCrCb]: ./output_images/car_YCrCb.png
[hog]: ./output_images/hog.png
[sliding_windows]: ./output_images/sliding_windows.png
[hot_windows]: ./output_images/hot_windows.png
[heat_map]: ./output_images/heat_map.png
[car_positions]: ./output_images/car_positions.png
[project_video]: ./project_video.mp4
[tracking_video]: ./tracking.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell #2-#6 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images (code cell #2).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car][car]
![notcar][notcar]

The HOG wrapper around `skimage.hog()` can be found in cell #3 of the IPython notebook. Among other things, it lets us handle returning both the feature vector and the 2D image representation of the HOG features, as well as the typical case where the number of pixels per cell and cells per block are provided symmetrically in the `x` and `y` dimensions.

Since I determined through earlier experiments that other features such as spatial binning and color histograms would be helpful as features as well, I implemented them in cells #4 (spatial binning) and #5 (color histogram). The spatial binning wrapper just calls resize, but we need to flatten the feature vector, so it's better to abstract it away. The histogram wrapper takes care of the fact that each channel needs to be processed separately, and the feature vector needs to be flat again, so the concatenated results are returned.

The featurization function that calls the above convenience wrappers is implemented in cell #6 of the IPython notebook. We simply accept the image path along with parameters such as the chosen color space, the number of histogram bins, etc. The image that is loaded is first loaded as RGB, so if we chose a different color space, we need to change the representation. If spatial binning was chosen (boolean argument to the function), we compute the spatial binning using the previously mentioned convenience function. We do the same with color histograms, which we can coose to select or not. Finally, we run the HOG feature extraction. Once we generate all the features, they end up being concatenated and returned. Note that since they may be on different scales, we'll need to use feature scaling. I will discuss this later. 

Here is an example of one of the above car image from the SVM training set being transformed into a HOG representation.

![hog][hog]

The transition from RGB to YCrCb, which is the color space I ultimately used, decidedly makes the images look odd to a human, but this color space is typically not for display purposes. It was originally conceived to allow for better representation of luminance (Y), so that it could be given more bandwidth than the chroma red-different (Cr) and chroma blue difference (Cb). However, it's also a useful transformation for machine learning, because we can focus on luminance, which is typically good enough for us to discern shapes, and the color information is treated separately.

![car_YCrCb][car_YCrCb]

#### 2. Explain how you settled on your final choice of HOG parameters.

In cell #7, we finally make use of all of the previously mentioned infrastructure. We take the car and non-car file names and generate the features. Note that at this point we need to select the color space, spatial size, etc. These selections were made during a very long tuning process. Interestingly, I chose other selections earlier that improved the validation accuracy of the SVM, but ended up doing more poorly together with the post-processing of classification, such as the heat map-based selection of final bounding boxes. So, I went through many cycles of feature engineering, SVM hyperparameter tuning (e.g. the regularization parameter), and postprocessing parameter tuning. Therefore, it would be hard to pinpoint exactly why I ended up choosing the parameters that I am about to mention here, since they were selected based on the final feedback on the test video, as opposed to merely on the validation accuracy of the SVM. 

That said, we can have some basic intuitions as to why the YCrCb color space was more useful than RGB, for example - luminance (Y) differences are more critical for object identification than the actual color (we could make a similar argument about the L channel in HSL, or V in HSV). Also, for both "color" channel binning and HOG features, it's good to separate some sense of color (Cr and Cb in YCrCb, or H in HSV/HSL) from other visual aspects (saturation in HSV, luminance in YCrCb). This separation is intuitively more useful than raw RGB intensities.

We can make similar arguments about keeping the number of pixels per cell fairly small (16) for the HOG computation, but using blocking (2 cells per block) to smooth out the histograms. I didn't think that HOG features could benefit from all three YCrCb channels, my hypothesis was that only using the Y channel would be adequate, but both the SVM validation accuracy and the postprocessed results visible in the video ended up being better with all three channels going into the HOG feature calculation.

| Parameter  | Value  |
|---|---|
| color space  | YCrCb   |
| spatial size  | (16, 16)   |
| orient  | 11   |
| pixels per cell  | 16   |
| cells per block  | 2   |
| HOG channels  | ALL (i.e. 3)   |
| histogram bins  | 64   |
| orient  | 11   |
| spatial features  | true   |
| histogram features  | true   |
| HOG features  | true   |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training the SVM, I scaled the features in cell #8 of the IPython notebook. Unscaled features cause disparities in feature importance, so it's critical to bring all features into the same range. It's also critical that we split the data into the training and validation sets, since the training set accuracy is almost always overly optimistic about the predictive ability of the model - we can easily overfit the training data and fail to generalize on new data, but it's the prediction on new data that's our ultimate goal. Frankly, even validation set accuracy is often higher than test set accuracy, because we keep changing the model (e.g. linear vs. RBF kernel SVM), the features, etc. to improve validation accuracy, so the model indirectly "sees" the validation data because it keeps being tuned to validation data. Still, using validation data, even without test data at first, is a good start, be it for manual model tuning, or hyperparameter sweeps (e.g. using an automated grid search or random search to find the best regularization parameter).

I actually tried the RBF kernel SVM in addition to the linear SVM, and it produced a model that was about 1 percentage point more accurate on the validation set, however it took much longer to train, the inference was much slower, and given all the other "knobs" such as the sliding window settings for postprocessing, using the RBF kernel resulted in worse final bounding boxes, so I kept the linear SVM. However, even for the linear SVM, there were some things to tune, especially the regularization parameter.

In cell #9 of the IPython notebook, we can see that I selected a linear SVM with an L2 regularization penalty and a regularization parameter (C) value of 0.001. I also chose the squared hinge loss. I tried L1 penalty instead of L2 as well, which is a common way to make the model more sparse by forcing some parameter values to zero. This tends to help when we have too many features, however here the L1 penalty did worse tan L2. The best validation accuracy that I obtained, given the features that I chose (and the parameters used to produce them) and the choice of the linear SVM, was 0.989.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search can be broken up into several problems. First, we need to decide what x and y ranges do we want to search. In addition to efficiency considerations (the smaller the regions, the more images we'll be able to process per second), we don't want to search certain parts of the image, in order to avoid false positives. For example, searching the sky above the road won't give any tracking benefits, and if anything, will generate false positives. Once we have our x and y ranges to search over, and once we decide what the window overlap should be (e.g. are they mutually exclusive, non-overlapping windows, or is there an overlap of say 50% between neigboring windows of the same size), we can run the slide_window() function in cell #10 of the IPython notebook. This function will simply provide the window boundaries, generated based on the above mentioned regions and overlap selections.

![sliding_windows][sliding_windows]

Once we have the sliding windows, we need to take the crops which given windows cover, feed them into the SVM, and see which ones qualified as positive classifications (i.e. that this window was recognized as having a car in it). This is done by the search_windows() function in cell #13 of the IPython notebook. That function extracts the feature for a given window (call to single_img_features() function in cell #11), scales the features, and applies the classifier. Only when a prediction is positive (an expectation that it is a car) do we add this window to the list of windows that contain likely detections. Notice how there are way fewer "hot" windows in the image below than there are of all windows in the image above.

![hot_windows][hot_windows]

Since the "hot" windows will cover the same area at different scales, or will cover neighboring parts of a given car, we need to merge bounding boxes that touch or overlap each other. To do this, we first look at the heat map that covers the number of times a given pixel was recognized as belonging to a car

![heat_map][heat_map]

The heat map above is generated in cell #14. Once we have the heat map, we can check if the number of overlapping or multi-scale occurrences passed the threshold desired to reduce the probability of false positives (code cell #15). Also, the heat map will delineate the boundaries of the union of the overlapping window boundaries. 

The pipeline is wrapped in the Pipeline class, the `__call__()` method of which is used to process individual video frames. Instead of having just a function, we need a class or a generator to keep track of the state, because one of the ways to fix the false positive problem is to keep track of detected windows in adjacent frames, and to filter out detections that don't overlap with other detections in the N (e.g. 15) adjacent frames. We could do it using a global variable, but that's not a good programming practice. Also, a generator would probably be suboptimal here. It's easier to create multiple instances of the same class, which keeps track of the state, e.g. if we need multiple instances for testing. The Pipeline class keeps the bounding box history in a field called `box_history`.  This is a moving window of N (e.g. 15) frames - once we saturate the "history," adding a new video frame causes the oldest video frame's bounding boxes to be removed. The Pipeline class's `__call__` method applies the previously mentioned pipeline, calling `search_windows()`, `add_heat()` and so on, and producing the final image annotated with the predicted vehicle bounding boxes.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

