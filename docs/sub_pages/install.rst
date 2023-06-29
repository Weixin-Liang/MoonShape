Code and Data: Install
===============================================


.. raw:: html
   
   <i class="fa fa-github"></i> View on and Install via <a
   href="https://anonymous.4open.science/r/moonshape_subpopulation/README.md">moonshape_subpopulation</a> 
   <br /> <br />

.. <i class="fa fa-github"></i> View on and Install via <a
.. href="https://anonymous.4open.science/r/Modality-Gap-UAI2022/">Anonymous GitHub.</a> 
.. <br /> <br />



Repo Structure Overview
-----------------------

.. code:: plain

   .
   ├── README.md
   ├── main.py
   ├── utils.py
   ├── figures/
   ├── datasets/
   │   └── metashift/metashift_prepare.ipynb
   │   └── modified-cifar4/cifar4_prepare.ipynb
   │   └── officehome/officehome_prepare.ipynb
   │   └── pacs/PACS_prepare.ipynb
   │   └── waterbireds/waterbireds_prepare.ipynb


We organize the code in the orders of the figures as presented in the
paper. As the folder name indicated, the `datasets`
folder provides the code for generating datasets.


Citation
--------

.. code-block:: bibtex

   @misc{liang2023accuracy,
         title={Accuracy on the Curve: On the Nonlinear Correlation of ML Performance Between Data Subpopulations}, 
         author={Weixin Liang and Yining Mao and Yongchan Kwon and Xinyu Yang and James Zou},
         year={2023},
         eprint={2305.02995},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }