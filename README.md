#Inria Aerial Image Labeling
##Building Footprint Extraction using Deep Semantic Segmentation 

To be able to use the code please follow listed instructions:

1)  Fill in the form and download data from https://project.inria.fr/aerialimagelabeling/download/
 
2)  Extract downloaded files and place them into *data* folder using the following folder structure:
    ```    
    data/test/images/*.tif
    data/train/images/*.tif
    data/train/gt/*.tif
    ```  

3) Execute *prepare_data.py* to image patches needed for training. The result would be the following folder structure:  
   ```
   data/train_384x384/images/*.jpg
   data/train_384x384/gt/*.png
   ```

4) Execute *train.py* to initially train all 6 models. In case of an out of memory problem, adjust *batch size* in *settings.py*:  
   ```
   batch_size = 9
   ```
   
5) Execute *fine_tune.py* to fine tune all 6 models. In case of an out of memory problem, adjust *batch size* in *settings.py*:  
   ```
   batch_size = 9
   ```

6) Execute *evaluate.py* to evaluate fine-tuned models. The results will be placed in:
   ```
   tmp/eval_ft_1/*
   tmp/eval_ft_2/*
   tmp/eval_ft_3/*
   tmp/eval_ft_4/*
   tmp/eval_ft_5/*
   tmp/eval_ft_6/*
   ```

7) Execute *prepare_submission.py* to generate grayscale predictions for the test images. The results will be placed in:
   ```
   tmp/submission_grayscale/*
   ```

8) Execute *grayscale_to_submission.py* to prepare contest submission (requires GDAL). The results will be placed in:
   ```
   tmp/submission_0.45/*
   tmp/submission_0.45.zip
   ```
