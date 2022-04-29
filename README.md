# image-extract-features
For given set of images (grayscale and color). we can extract the unique features in all images using Harris operator and λ-, generate feature descriptors using scale invariant features (SIFT), and matching the image set features using sum of squared differences (SSD) and normalized cross correlations.
# Feature matching 
## a) SSD:
### this algorithm is dependent on minimizing the error ( Difference ) between pixels, so the lower the value the higher the matching.
## b) NCC:
### this algorithm is the opposite to the prev. one, it is dependent on maximizing the correlation between pixels, so the higher the correlation the higher the matching.
## Results for Feature matching
### there are 2 params. for the gui in this part, selecting the mode( "NCC","SSD" ), and select the threshold(for SSD: [0,inf[, for NCC: [-1,1] )
### for our test case we used threshold 10000 for SSD, and 0.9 for NCC.
![ NCC Image with 0.9 threshold](results/ncc.png)

![ SSD Image with 10000 threshold](results/ssd.png)
