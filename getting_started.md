(In progress... , for now you can only test classification algorithm from this part of documentation but recognition/detection documentation part is yet to be completed)     
## Installation :     
Tip : Please make sure to install all dependencies from [requirements.txt](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/requirements.txt).       
Grab a cup of coffee as these will take some time !      
## Get, set and go :        
* Download complete Project files using following command from git bash:       
```
git clone https://github.com/souravs17031999/Dog-Breed-Classifier-App      
```     
* Open terminal (cmd) and then move to Project root directory.   
* Now, run the following file "[dog_breed_predict.py](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/dog_breed_predict.py)" using following command (which also gives path of image supplied) :     
```
python dog_breed_predict.py <PATH>
```  
where python (python3) depending upon env and <PATH> is simply the image actual path you supply (without "<", ">").    
Ex.  
 
``` 
python dog_breed_predict.py dog.JPG
```    
* There you go, predicted label name along with the image you supplied will be printed on python console.  

#### For live predictions :     
* IMPORTANT : If you want to run live predictions from feed taken from your live cam, then instead of file "dog_breed_predict.py" which gives predictions on input images, you should run "[dog_breed_predict_cam.py](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/dog_breed_predict_cam.py)"     

NOTE : you should already have weights of pre trained model used in this project (mail me at souravs_1999@rediffmail.com for model_transfer.pt file) and replace the following line with your model_transfer.pt file path and also you need "resnet50.pth" file which is already available [here at kaggle](https://www.kaggle.com/pytorch/resnet50) (download this).     
```
model_transfer = load_model('model_transfer.pt')   
```   
TIP : Keep everything in the same project folder and enjoy !     
