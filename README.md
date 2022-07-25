# DeepFPC
This repository provides the implementation of DeepFPC from the following paper:

Model-Informed unsupervised Deep Learning Approach to Frequency and Phase Correction of MRS Signals: [Link to arxiv](https://www.biorxiv.org/content/10.1101/2022.06.28.497332v1)

## How does it work?
- DeepFPC was implemented in Python with the help of the Pytorch lightning interface. 
- For each experiment, a "run" json file should be created. All parameters of the deep neural network and data can be stated in the json file.
There is examples of "run" json files that can be found in the "runs" folder.
- The network can be trained and tested simply by running "main.py". 
- Engine.py controls the pre and post-training steps for training and testing. dotrain() and dotest() are two functions for training and testing modes, respectively.
- Model.py is an object inherited from PyTorch lightning's "LightningModule". Now it contains two neural networks (ConvNet & MLPNet), but you can easily add your model. Deep spectral registration model and Deep Cr referencing model are implemented as dCr() and dSR() functions. 
------
## Proposed Deep Autoencoder for Deep Learning-based Peak Referencing
|![img_1.png](images/Figure%202.png)|
|:--:|
|Illustration of the proposed convolutional encoder–model decoder for dCrR method. |
------
## Result
### Phantom
|![img.png](images/Figure%205.png)|
|:--:|
|Frequency and phase correction of the phantom test subset using dCrR method. Uncorrected (a) and corrected (b) spectra from the test subset. The circled inset show zoomed Cr peak at 3 ppm. The similarity matrix of 64 samples of the test subset before (c) and after (d) FPC. dCrR, deep learning-based Creatine referencing; LW, linewidth.|
### GABA-edited in-vivo dataset([Big GABA](https://www.nitrc.org/projects/biggaba/))
|![img.png](images/Figure%207.png)|
|:--:|
|An example of FPC using dCrR for a test set in the GABA-edited in-vivo dataset. Unedited spectra (a) and their similarity matrix (b) before FPC. Edited spectra (c) and their similarity matrix (d) before FPC. Unedited spectra (e) and their similarity matrix (f) after FPC. Edited spectra (g) and their similarity matrix (h) after FPC. (i) Average uncorrected spectra (blue, unedited; red, edited) and their difference (dark green). (j) Average corrected spectra using dCrR (blue, unedited; red, edited) and their difference (dark green) dCrR, deep learning-based Creatine referencing.|
-----
## Acknowledgments
This project has received funding from the European Union's Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No 813120.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@article {Shamaei2022.06.28.497332,
	author = {Shamaei, Amir Mohammad and Starcukova, Jana and Pavlova, Iveta and Starcuk, Zenon},
	title = {Model-Informed Unsupervised Deep Learning Approaches to Frequency and Phase Correction of MRS Signals},
	elocation-id = {2022.06.28.497332},
	year = {2022},
	doi = {10.1101/2022.06.28.497332},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Purpose: A supervised deep learning (DL) approach for frequency-and-phase Correction (FPC) of MR spectroscopy (MRS) data recently showed encouraging results, but obtaining transients with labels for supervised learning is challenging. This work investigates the feasibility and efficiency of unsupervised DL-based FPC. Method: Two novel DL-based FPC methods (deep learning-based Cr referencing [dCrR] and deep learning-based spectral registration [dSR]) which use a priori physics domain knowledge are presented. The proposed networks were trained, validated, and evaluated using simulated, phantom, and publicly accessible in-vivo MEGA-edited MRS data. The performance of our proposed FPC methods was compared to other generally used FPC methods, in terms of precision and time efficiency. A new measure was proposed in this study to evaluate the FPC method performance. The ability of each of our methods to carry out FPC at varying SNR levels was evaluated. A Monte Carlo (MC) study was carried out to investigate the performance of our proposed methods. Result: The validation using low-SNR manipulated simulated data demonstrated that the proposed methods could perform FPC comparably to other methods. The evaluation showed that the dCrR method achieved the highest performance in phantom data. The applicability of the proposed method for FPC of GABA-edited in-vivo MRS data was demonstrated. Our proposed networks have the potential to reduce computation time significantly. Conclusion: The proposed physics-informed deep neural networks trained in an unsupervised manner with complex data can offer efficient FPC of MRS data in a shorter time.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/07/03/2022.06.28.497332},
	eprint = {https://www.biorxiv.org/content/early/2022/07/03/2022.06.28.497332.full.pdf},
	journal = {bioRxiv}
}

```
