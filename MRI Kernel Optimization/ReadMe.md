The following code represents multiple parallel computing approaches used to iteratively solve the Q and FhD components to identify the MRI voxel values through the equation :

                                (FhF + λWhW)ρ=FhD

Where the (FhF + λWhW) component is the pre-computed Q vector, ρ is the unknown vector of MRI voxel values, Fh is a matrix that models the physics of the imaging process, and D is the sample data from the MRI scanner.

The goal of this parallel computing approach comparison is to generate a parallel computing algorithm to parse through the available data (in unit millions) and calculate the Q and FhD components in as little time as possible. 

Identifying an efficient algorithm for calculating the MRI voxels is imperative to providing real-time medical assessments to patients and provide treatments as soon as possible.

The following code is a collection of different kernels utilizing experimental methods for voxel computations, data collection and pre-processing has been performed separately. The best results given were with Kernel 5 - Special Function Units (SFU).

The image shown below is a visual representation of the iterative algorithm approach:

![image](https://user-images.githubusercontent.com/80390906/231824788-ce4db252-bfd0-4d6c-ad5f-52f547653d2c.png)

We then compared the results of our produced image to the sample image provided by the MRI dataset.
The validation approach is as follows:
![image](https://user-images.githubusercontent.com/80390906/231825479-6a5f748f-1725-4a3f-b7a5-7fdd4d1b68b8.png)

The images produced and error percentages are as follows:
![image](https://user-images.githubusercontent.com/80390906/231824622-0d9e96e4-4edb-4a45-93d1-e787b43ddb9a.png)

The computational speedups are shown below, we can see that the SFU kernel with constant memory access performs the best, providing upto 108x sppedup in overall reconstruction time as compared to the sequential approach.

![image](https://user-images.githubusercontent.com/80390906/231826201-6da7e010-aced-4f3d-9489-50a6ae18468f.png)

