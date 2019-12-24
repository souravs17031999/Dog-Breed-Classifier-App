## Installation :     
Tip : Make sure to install [Numpy](https://pypi.org/project/numpy/), [Pandas](https://pypi.org/project/pandas/), [Matplotlib](https://pypi.org/project/matplotlib/) first and then proceed next.     
* [Torch package](https://pytorch.org/get-started/locally/)    
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
where python (python3) depending upon env and <PATH> is simply the immage actual path you supply (without "<", ">").    
Ex.  
 
``` 
python dog_breed_predict.py dog.JPG
```    
* There you go, predicted label name along with the image you supplied will be printed on python console.  

NOTE : Please sure to install all dependencies from [requirements.txt](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/requirements.txt) and you should already have weights of pre trained model used in this project (mail me at souravs_1999@rediffmail.com for model.pt file).    
