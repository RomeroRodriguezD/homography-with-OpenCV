# Homography with OpenCV #

Using perspective transformation and masking to put an image inside another. This could be easily done by catching the reference points of both images and then applying the right operations. 
<br>
<br>
-First, we'll need to generate 2 matrices, one for each image and its key points.
<br>
-Then we'll need to calculate the homography matrix and warp the input image to the destination image points.
<br>
-Finally, we'll need a mask that will be used to mix both images.

**Creating the source matrix.**

Since this one is all about height and shape, we won't need to define the points manually on this one.


```python
# Source & Destination image paths.

dst_img_path = './times_square.jpg'
src_img_path = './sonic.jpg'

# Loads and captures source image shape.

im_src = cv2.imread(src_img_path)
(srcH, srcW) = im_src.shape[:2]

# Generate a source matrix, we'll only need height and width.

pts_src = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
```

**Creating the destination matrix**

This one will require to select the 4 corner coordinates manually, following this order: 

-Top left.
<br>
-Top right.
<br>
-Bottom right.
<br>
-Bottom left.

For this we can use a mouse callback that will store up to 4 points.


```python
%matplotlib notebook

import matplotlib.pyplot as plt

pts_dst = []

def click_event_IDE(event, x, y, flags, param):
    """This version of the function only applies to OpenCV used on IDE's, not in Jupyter Notebook / Google Colab."""
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_dst.append((x,y))
        print(f'Coordinates: ({x}, {y})')
        if len(pts_dst) == 4:
            cv2.destroyAllWindows() # Close the image after 4 points selected

def click_event_NB(event):
    """This function, used combined with Matplotlib, will let us display the image and use the callbacks."""
    if len(pts_dst) < 4:
        pts_dst.append((event.xdata, event.ydata))
        #print(f'Coordinates: ({event.xdata}, {event.ydata})') You can't use matplotlib + print in Notebook
    else:
        #print('You have already stored 4 coordinates, move on to the next cell.')
        plt.close()

        
im_dst = cv2.imread(dst_img_path)
imgH, imgW = im_dst.shape[:2]

plt.imshow(cv2.cvtColor(im_dst, cv2.COLOR_BGR2RGB))
#plt.axis('off')  # Disable axis. I'll rather keep it just to see the coordinates

# Connection the "on click" event to tracj coordinates.
plt.connect('button_press_event', click_event_NB)

```

![image](https://github.com/RomeroRodriguezD/homography-with-OpenCV/assets/105886661/86cb7848-8ad6-437c-9ae6-2955091c6965)


**Checking the stored coordinates and turning them into Numpy array**


```python
pts_dst = np.array(pts_dst)
print(pts_dst)
```

    [[ 70.41733871 133.64717742]
     [173.64314516 223.96975806]
     [154.28830645 275.58266129]
     [ 18.80443548 206.22782258]]
    

**Calculate homography**

Homography implies mapping our coordinates from the source image, to the destination one, so they can be aligned, correcting the perspective distortion. But in this case it will be used to replace part of the destination image.


```python
# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
```

**Masking**

Now, we'll need a mask to copy our source image into the destination one.

```python
mask = np.zeros((imgH, imgW), dtype="uint8")
cv2.fillConvexPoly(mask, pts_dst.astype("int32"), (255, 255, 255), cv2.LINE_AA) # LINE_AA, antialiasing, so they're thin and smooth

# We re-scale the mask so it goes from 1 color channel to 3 color channel, by stacking it three times, since the original image
# is an RGB spectrum image.

maskScaled = mask.copy() / 255.0 # First we divide each color value and set them to 1.0 or 0.0
maskScaled = np.dstack([maskScaled] * 3) # Then we triple the dimensions of the mask so it can match the RGB image
```

Here it goes:

![image](https://github.com/RomeroRodriguezD/homography-with-OpenCV/assets/105886661/231d330b-61af-4257-bad6-34ad3e812a29)

**Copying the warped source image into the input image**

This is a three steps process:

1-Multiplying the warped image (the one that will be placed inside the other) and the scaled mask, together.
<br>
2-Multiplying the original destination image with the scaled mask.
<br>
3-Adding both as output, and casting the result into "int8".
<br>

Multiplications on the warped image are done so it fills the white space of the mask, while multiplications on the destination image makes it fill the black space of the mask.


```python
warpedMultiplied = cv2.multiply(im_out.astype("float"), maskScaled)
imageMultiplied = cv2.multiply(im_dst.astype(float), 1.0 - maskScaled)
output = cv2.add(warpedMultiplied, imageMultiplied)
output = output.astype("uint8")
```

**Source image that will be place on the destination**


```python
plt.imshow(cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB))
```

![image](https://github.com/RomeroRodriguezD/homography-with-OpenCV/assets/105886661/9abd1be1-696b-4e81-9cef-bd5e01d66fb8)




**Check the result**

Now Sonic The Hedgehog from 1991 it's part of the Times Square billboard.


```python
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
```

![inserted_sonic](https://github.com/RomeroRodriguezD/homography-with-OpenCV/assets/105886661/691e8712-72aa-4089-9a11-6f6457778703)


