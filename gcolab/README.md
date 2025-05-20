### Instructions for Google Colab

In order to run our notebooks in Google colab, you have to run the following commands in a gcolab kernel

```
!pip install -q condacolab
!pip install git+https://github.com/antonyvam/nDTomo.git
import condacolab
condacolab.install()
```

Now we need to wait for the kernel to restart. Then, we install the astra-toolbox.

```
!conda install -c astra-toolbox -c nvidia astra-toolbox
```

