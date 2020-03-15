# Radar-Target-Generation-and-Detection
Udacity Project Submission: Radar Target Generation and Detection
<img src="image/image11.png" width="700" height="400" />

Udacity Sensor Fusion course for self-driving cars.
In this course we will be talking about sensor fusion, whch is the process of taking data from multiple sensors and combining it to give us a better understanding of the world around us. we will mostly be focusing on two sensors, lidar, and radar. By the end we will be fusing the data from these two sensors to track multiple cars on the road, estimating their positions and speed.

Radar data is typically very sparse and in a limited range, however it can directly tell us how fast an object is moving in a certain direction. This ability makes radars a very pratical sensor for doing things like cruise control where its important to know how fast the car infront of you is traveling. Radar sensors are also very affordable and common now of days in newer cars.

### Final Radar Project Rubrix
In radar module we learned about Radar Priciples, FMCW Waveform Design, Range, Velocity and Angular Calculations.
Frequency domain analysis of Radar signal using 1D FFD for Range calculations and 2D FFD for doppler range calculation.
Calculation of Noise level using CA CFAR in 1D and 2D.

**Implementation steps for the 2D CFAR process.**
1) Determine the number of Training cells for each dimension Tr and Td from RDM. Similarly, pick the number of guard cells Gr and Gd.
2) Slide the Cell Under Test (CUT) across the complete cell matrix (Range x Doppler)

        for curr_range = Tr+Gr+1: S1-(Tr+Gr)
            for curr_doppler = Td+Gd+1: S2-(Td+Gd)
            
3) Determine the signal level at the Cell Under Test.

    CUT = db2pow(RDM(curr_range,curr_doppler));% cell under test
    
4) Sum of all cells in Training window
5) Sum of All cells in Gaurd window
6) Measure and average the noise across all the training cells. This gives the threshold

    AvgNoise = (SumAllcells - SumGuardcells - CUT)/TotalTrainingCells;
    
7) Add the offset (if in signal strength in dB) to the threshold to keep the false alarm to the minimum.
8) If the CUT signal level is greater than the Threshold, assign a value of 1, else equate it to zero.

     if (RDM(curr_range,curr_doppler)>(noise_level(curr_range,curr_doppler)+offset)
                RDM_out(curr_range,curr_doppler) = 1;
      else
                RDM_out(curr_range,curr_doppler) = 0; 
      end

**Selection of Training, Guard cells and offset.**
  Training cells for each dimension Tr and Td from RDM and by looking at doppler plot
  Similarly, pick the number of guard cells Gr and Gd based on maximum range of doppler shift and velocity resolution.
  
**Steps taken to suppress the non-thresholded cells at the edges.**
% % Since the cell under test are not located at the edges, due to the training cells occupying the edges, we suppress the edges to zero. Any cell value that is neither 1 nor a 0, assign it a zero.
  - Output matrix of size of RDM is initalized with all 0.
