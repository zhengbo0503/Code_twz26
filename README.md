# Code_twz26
Version: December 19, 2025

## Usage 
These codes has been tested in the following configuration:
```
OS: macOS 26.0.1 25A362 arm64 
CPU: Apple M3 Pro 
GPU: Apple M3 Pro 
Memory: 36864MiB 
MATLAB version: 25.2.0.3042426 (R2025b) Update 1
```

## Requirements 
To run the code, you need 
 - [Advanpix Multiprecision Computing Toolbox](https://www.advanpix.com/)
   to simulate quadruple precision in MATLAB,
 - [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) to compile a MEX
   file for `DGESVJ` and `DGEJSV`.
   
## Codes description
 - The main algorithm is
   [`mposj.m`](https://github.com/zhengbo0503/Code_twz26/blob/main/mposj.m)
   which implements our algorithm with (single, double, quadruple) setting. 
 - Another implementation of our algorithm using (single, single, double)
   setting is
   [`mposj_ssd`](https://github.com/zhengbo0503/Code_twz26/blob/main/mposj_ssd.m). 
 - `testx.m` are different tests for our paper. 
