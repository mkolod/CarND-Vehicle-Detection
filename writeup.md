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
[hog]: ./examples/hog.png
[sliding_windows]: ./output_images/sliding_windows.jpg
[hot_windows]: ./output_images/hot_windows.jpg
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

In cell #7, we finally make use of all of the previously mentioned infrastructure. We take the car and non-car file names and generate the features. Note that at this point we need to select the color space, spatial size, etc. These selections were made during a very long tuning process. Interestingly, I chose other selections earlier that improved the validation accuracy of the SVM, but ended up doing more poorly together with the post-processing of classification, such as the heat map-based selection of final bounding boxes. So, I went through many cycles of feature engineering, SVM hyperparameter tuning (e.g. the regularization parameter), and postprocessing parameter tuning. Therefore, it would be hard to pinpoint exactly why I ended up choosing the parameters that I am about to mention here, since they were selected based on the final feedback on the test video, as opposed to merely on the validation accuracy of the SVM.

color_space = 'YCrCb' 
spatial_size = (16, 16)
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' 
hist_bins = 64
spatial_feat=True
hist_feat=True
hog_feat=True

I then explored different color spaces and different HOG `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

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

