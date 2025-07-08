
## Import libraries
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, PredefinedSplit, GroupShuffleSplit
from sklearn import preprocessing
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor, Normalize
import torchvision.models as models
import timeit
import csv
import random

from collections import Counter

# RAM130521:1944 HABRÍA QUE INTENTAR NO USAR DROP CON LOS INDICES EN EL EJE 0, SINO LOC BOOLEANO, COMO CON RF

# RAM190321: Reproducibility
# RAM240222: Hasta hoy estaba con semilla=10. He puesto semilla=0
semilla=0
random.seed(semilla)
torch.manual_seed(semilla)

# Set the server to compute with GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

## Load the data and set index
images = pd.read_pickle('/home/commondata/imageset.pickle')
types = pd.read_pickle('/home/commondata/metadata.pickle')
types.reset_index(inplace=True)

# heinle = pd.read_csv('/home/aler/vlavado_nubes/Random_Forest_RAM/featuresCam1.txt', sep=" ")
# heinle.reset_index(inplace=True)
# heinle['datetimes'] = heinle['level_1'] + ' ' +heinle['datetimes']
# auxcols = ['level_1', 'level_0']
# heinle = heinle.drop(labels=auxcols, axis=1)

## Preprocess the data

# Data types

type(images)
type(types)
# type(heinle)

# Check Images dimension
images.shape

# Select Camera 1 images for processing

images1 = images[:,:,:,0,:]

images1.shape

# Change the RGB axis to visualize images1

images1 = np.moveaxis(images1,0,-2)

plt.imshow(images1[:,:,:,0])
plt.show()

# Change array dtype to uit8

images1 = images1.astype(np.uint8)

# Inspect first rows of types and Heinle attributes
types.head()
types['datetimes']
# heinle.head()

# Obtain the column names of types of clouds and drop aerosol, multinube and nimboestrato

types.columns # Column 'types' contains the type of cloud
types["types"]

# aerosol_idx = types.index[types['types'] == 'aerosol'].tolist() # index 205 has aerosol
# multinube_idx = types.index[types['types'] == 'multinube'].tolist() # index 448 has multinube
# nimboestrato_idx = types.index[types['types'] == 'nimboestrato'].tolist() # index 808 has nimboestrato

# check the index obtained are right by using iloc

# types.iloc[205] # matches with aerosol
# types.iloc[448] # matches with multinube
# types.iloc[808] # matches with nimboestrato

# The list of index is fine.

#Let's drop all the rows with those indexes

# Start by creating a copy of the data

clouds = types.copy(deep=True)

# clouds = clouds.drop(labels=aerosol_idx, axis=0)
clouds = clouds.loc[clouds.types != 'aerosol']

drop_multinube = True
if(drop_multinube):
   # clouds = clouds.drop(labels=multinube_idx, axis=0)
    clouds = clouds.loc[clouds.types != 'multinube']

# clouds = clouds.drop(labels=nimboestrato_idx, axis=0)
clouds = clouds[clouds.types != 'nimboestrato']

# Check how many rows are left

clouds.shape # (988, 37)

# Compare it with the original dataset
# len(types["types"]) - len(aerosol_idx) - len(multinube_idx) - len(nimboestrato_idx)
# 988 rows after dropping the rows

# Check there is no aerosol, multinube and nimboestrato in the clouds dataset

clouds['types'].str.contains('aerosol').any()
clouds['types'].str.contains('multinube').any()
clouds['types'].str.contains('nimboestrato').any()

# Drop several Heinle attributes attributes
# These attributes are the ones obtained by the ceilometer

# ceilometer = ["cloudLayers", "cbh_l1", "cbh_l2", "cbh_l3", "cdp_l1", "cdp_l2", "cdp_l3"]
# heinlered = heinle.copy(deep=True)
# heinlered = heinlered.drop(labels=ceilometer, axis=1)

# Drop the rows corresponding to aerosol, multinube, nimboestrato

# heinlered = heinlered.drop(labels=aerosol_idx, axis=0)
# if(drop_multinube):
#     heinlered = heinlered.drop(labels=multinube_idx, axis=0)
# heinlered = heinlered.drop(labels=nimboestrato_idx, axis=0)

# heinlered.shape # (988, 15)

# Compare it with the original dataset
# len(heinle["types"]) - len(aerosol_idx) - len(multinube_idx) - len(nimboestrato_idx)
# 988 rows after dropping the rows

# Check there is no aerosol, multinube and nimboestrato in the heinlered dataset

# heinlered['types'].str.contains('aerosol').any()
# heinlered['types'].str.contains('multinube').any()
# heinlered['types'].str.contains('nimboestrato').any()

# Finally, it is necessary to drop the images corresponding to the aerosol, multinube and nimboestrato

# Turn the list of index to arrays
# aerosol_arr = np.asarray(aerosol_idx)
# multinube_arr = np.asarray(multinube_idx)
# nimboestrato_arr = np.asarray(nimboestrato_idx)

# if(drop_multinube):
#     tot_array = np.concatenate((aerosol_arr, multinube_arr, nimboestrato_arr))
# else:
#     tot_array = np.concatenate((aerosol_arr, nimboestrato_arr))

img1 = np.copy(images1)


# img1 = np.delete(img1, tot_array, axis=-1)
img1 = img1[..., clouds.index.values]

# np.delete(img1, tot_array, axis=-1)

# Check if the number of images in the array is 988
img1.shape # (500, 500, 3, 988)

# Check if there are any missing values

# print(heinlered.isna().any())
print(clouds.isna().any())

## Create a stratified partition for training, validation and testing datasets
# Let's take 60% of the whole dataset for training
# 20% for validation
# and 20% for testing

# First we get the index using StratifiedShuffleSplit from
# Scikit Learn and then we apply those index


# # First we split data into 60% training and 40% validation + testing
# split = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=10)
# for train_index, test_valid_index in split.split(clouds, clouds.types):
#     train_clouds = clouds.iloc[train_index]
#     test_valid_clouds = clouds.iloc[test_valid_index]


######## RAM170321: Voy a cambiar la estrategia de evaluación, para que no desordene los valores
# Si quiero usar StratifiedKFold, tengo que usar 2 folds, o sea que habrá 50% para test (en vez de 40%)
# Aparentemente lo que hace es, separa las clases, y después crea los folds. Con shuffle=False no debería desordenar

# split = StratifiedKFold(n_splits=3, shuffle=False, random_state=10)
# for train_valid_index, test_index in split.split(clouds, clouds.types):
#     break


# RAM180321: Nueva estrategia de evaluación. Asignamos un grupo a cada día, así las nubes del mismo día
# van a train o a test

start_date = clouds.datetimes.min()
end_date = clouds.datetimes.max()


# def first_half(x):
#     if(x<15): return('a')
#     elif(x<30): return('b')
#     elif(x<45): return('c') 
#     else: return('d')

# RAM210521:1759 Aquí voy a poner la nueva manera de seleccionar las muestras

if True:
    def first_half(x):
        if(x<30): return('a')
        else: return('b')    

    # from datetime import datetime as dt
    
    # dates = [x.date() for x in clouds.datetimes]
    # times = [x.time() for x in clouds.datetimes] 
    # dates_as_list = [dt.strptime(x[0]+' '+x[1], '%Y-%m-%d %H:%M:%S') for x in zip(dates,times)]



    # dates_as_list = [x.strftime('%Y-%m-%d %H:%M:%S') for x in clouds.datetimes]
    dates_as_list = clouds.datetimes
    dates_as_list_types = list(zip(dates_as_list, clouds.types.tolist()))

    # Groups every half hour
    kk = [str(x[0].year)+str(x[0].month).zfill(2)+str(x[0].day).zfill(2)+str(x[0].hour).zfill(2)+
            first_half(x[0].minute)+x[1] for x in dates_as_list_types]
    groups = np.array(kk)     

    # RAM150521:183 A cada grupo, le ponemos su clase
    groups_class = [(x, x[11:]) for x in groups]      

    # RAM190321: Reproducibility
    import random
    # RAM140521:1936 Originalmente semilla=10
    # semilla = 0
    random.seed(semilla)

    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, PredefinedSplit, GroupShuffleSplit
    
    split = StratifiedShuffleSplit(n_splits=1, train_size=.6, random_state=semilla)
    #split.get_n_splits()

    unique_groups = np.array(list(set(groups)))
    Xgroups = np.expand_dims(unique_groups, axis=1)
    ygroups = np.array([x[11:] for x in unique_groups])

    for train_valid_index_groups, test_index_groups in split.split(Xgroups, ygroups):
        break

    Xgroups_train_valid = Xgroups[train_valid_index_groups]
    ygroups_train_valid = ygroups[train_valid_index_groups]

    # RAM190321: Reproducibility
    random.seed(semilla)

    split2 = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=semilla)

    for train_index_groups, valid_index_groups in split2.split(Xgroups_train_valid, ygroups_train_valid):
        break

    groups_for_training = Xgroups_train_valid[train_index_groups]
    groups_for_valid = Xgroups_train_valid[valid_index_groups]
    groups_for_train_valid = Xgroups[train_valid_index_groups]
    groups_for_test = Xgroups[test_index_groups]

    train_valid_index = np.where(np.in1d(groups, groups_for_train_valid.squeeze()))[0]
    test_index = np.where(np.in1d(groups, groups_for_test.squeeze()))[0]

    train_index = np.where(np.in1d(groups, groups_for_training.squeeze()))[0]
    valid_index = np.where(np.in1d(groups, groups_for_valid.squeeze()))[0]

    # RAM 150521:2038 A partir de aquí es lo mismo que había
    train_valid_types = clouds.types.values[train_valid_index]
    test_types = clouds.types.values[test_index]
    train_valid_clouds = clouds.iloc[train_valid_index]
    train_valid_groups = groups[train_valid_index]

    # X of clouds, excluding datetimes (first column) and type (last column)
    Xclouds = clouds.iloc[:,1:-1].values  

    Xclouds_train_valid = Xclouds[train_valid_index, :]

    # test_heinlered = heinlered.iloc[test_index]
    test_groups = groups[test_index]

    train_types = clouds.types.values[train_index]
    valid_types = clouds.types.values[valid_index]

    train_groups = groups[train_index]
    valid_groups = groups[valid_index]

    # train_heinlered = heinlered.iloc[train_index]
    # valid_heinlered = heinlered.iloc[valid_index]

# RAM141021:1720 Esto es para visualizar las particiones
if(False):
    # RAM180521:1732 Quiero ver la proporción de datos en cada partición
    import collections
    def proporcion(y):
        muestra = collections.Counter(y)
        suma = sum(muestra.values())
        for x in muestra:
            muestra[x] /= suma
        muestra = pd.DataFrame.from_dict(muestra, orient='index')    
        return muestra

    from functools import reduce
    from pprint import pprint
    dfs = [proporcion(train_types),proporcion(valid_types), proporcion(test_types), proporcion(train_valid_types)  ]
    df_final = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), dfs)
    df_final.columns = ['Train', 'Val', 'Test', 'TrainVal']
    print('SEMILLA=',semilla)
    print('===========')
    print(df_final)




# RAM210521:1837 Así hacía la partición de datos antes
if(False): 
    def first_half(x):
        if(x<30): return('a')
        else: return('b')    

    dates_as_list = clouds.datetimes.tolist()
    dates_as_list_types = list(zip(dates_as_list, clouds.types.tolist()))

    # Groups every half hour
    kk = [str(x[0].year)+str(x[0].month).zfill(2)+str(x[0].day).zfill(2)+str(x[0].hour).zfill(2)+
                    first_half(x[0].minute)+x[1] for x in dates_as_list_types]
    groups = np.array(kk)                   

    # Groups every hour
    #kk = [str(x[0].year)+str(x[0].month).zfill(2)+str(x[0].day).zfill(2)+str(x[0].hour).zfill(2)+str(x[1]) for x in dates_as_list_types]
    #groups = np.array(kk)

    # Groups every day
    #kk = [str(x[0].year)+str(x[0].month).zfill(2)+str(x[0].day).zfill(2)+str(x[1]) for x in dates_as_list_types]
    #groups = np.array(kk)

    # Groups every hour, no types
    """
    kk = [str(x.year)+str(x.month).zfill(2)+str(x.day).zfill(2)+str(x.hour).zfill(2) for x in dates_as_list]
    groups = np.array(kk)
    """

    # RAM190321: Reproducibility
    random.seed(10)
    torch.manual_seed(10)

    #groups = np.arange(clouds.shape[0])
    split = GroupShuffleSplit(n_splits=100, train_size=.6, random_state=10)
    #split.get_n_splits()

    # X of clouds, excluding datetimes (first column) and type (last column)
    Xclouds = clouds.iloc[:,1:-1].values
    # Hay que asegurarse de que en train y en test están todos los tipos

    counter = 0
    for train_valid_index, test_index in split.split(Xclouds, clouds.types.values, groups):
        train_valid_types = clouds.types.values[train_valid_index]
        test_types = clouds.types.values[test_index]
        if(set(train_valid_types)==set(test_types)): 
            print(f'Encontrado en la iteración {counter}')
            break
        else:
            counter += 1
    else: 
        print('TrainValid/test: not found')                

    # Esto es para coger el primer split
    train_valid_clouds = clouds.iloc[train_valid_index]
    train_valid_groups = groups[train_valid_index]

    Xclouds_train_valid = Xclouds[train_valid_index, :]

    #test_valid_clouds = clouds.iloc[test_valid_index]
    test_clouds = clouds.iloc[test_index]
    test_groups = groups[test_index]

######## RAM170321: FIN

# We generate the same partitions for heinlered and img1

# train_valid_heinlered = heinlered.iloc[train_valid_index]
# test_heinlered = heinlered.iloc[test_index]

train_valid_img1 = np.take(img1,train_valid_index,-1)
test_img1 = np.take(img1,test_index,-1)

# # The 40% validation + testing is then splitted in half.
# # So you get 20% validation and 20% testing
# split2 = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=10)
# for test_index, valid_index in split2.split(test_valid_clouds, test_valid_clouds.types):
#     test_clouds = test_valid_clouds.iloc[test_index]
#     valid_clouds = test_valid_clouds.iloc[valid_index]


######## RAM170321: Y ahora hago lo mismo para validación / test
# Con el primero, desordenados
# split2 = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=10)

# Con el primero, desordenados
# split2 = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=10)
# Con el segundo, ordenados


# split2 = StratifiedKFold(n_splits=4, shuffle=False, random_state=10)

# for train_index, valid_index in split2.split(train_valid_clouds, train_valid_clouds.types):
#     break

# # RAM190321: Reproducibility
# random.seed(10)
# torch.manual_seed(10)

# split2 = GroupShuffleSplit(n_splits=100, train_size=.7, random_state=10)
# counter = 0
# for train_index, valid_index in split.split(Xclouds_train_valid, train_valid_types, train_valid_groups):
#     train_types = train_valid_types[train_index]
#     valid_types = train_valid_types[valid_index]
#     if(set(train_types)==set(valid_types)): 
#         print(f'Train/Valid: encontrado en la iteración {counter}')
#         break
#     else:
#         counter += 1
# else: 
#     print('Train / valid: not found')        

# train_groups = train_valid_groups[train_index]
# valid_groups = train_valid_groups[valid_index]

# train_clouds = train_valid_clouds.iloc[train_index]
# valid_clouds = train_valid_clouds.iloc[valid_index]

######## RAM170321: FIN


# We generate the same partitions for heinlered and img1

# train_heinlered = train_valid_heinlered.iloc[train_index]
# valid_heinlered = train_valid_heinlered.iloc[valid_index]

# train_img1 = np.take(train_valid_img1,train_index,-1)
# valid_img1 = np.take(train_valid_img1,valid_index,-1)

train_img1 = np.take(img1,train_index,-1)
valid_img1 = np.take(img1,valid_index,-1)


# Check the type of cloud is consistent among clouds and heinlered

# print(train_heinlered["types"] == train_clouds["types"])
# print(valid_heinlered["types"] == valid_clouds["types"])
# print(test_heinlered["types"] == test_clouds["types"])

# Change Series data type to tensor
# So we can process the labels in the dataloader

print('Check label encoding: ')
# train_clouds = train_clouds["types"].tolist()
# valid_clouds = valid_clouds["types"].tolist()
# test_clouds = test_clouds["types"].tolist()

train_valid_clouds = train_valid_types.tolist()
train_clouds = train_types.tolist()
valid_clouds = valid_types.tolist()
test_clouds = test_types.tolist()

# RAM180521:1732 Quiero ver la proporción de datos en cada partición
if(True):
    import collections
    def proporcion(y):
        muestra = collections.Counter(y)
        suma = sum(muestra.values())
        for x in muestra:
            muestra[x] /= suma
        muestra = pd.DataFrame.from_dict(muestra, orient='index')    
        return muestra

    def cuenta_absoluta(y):
        muestra = collections.Counter(y)
        suma = sum(muestra.values())
        for x in muestra:
            muestra[x] /= suma
        muestra = pd.DataFrame.from_dict(muestra, orient='index')    
        return muestra

    from functools import reduce
    from pprint import pprint
    dfs = [proporcion(train_clouds),proporcion(valid_clouds), proporcion(test_clouds), proporcion(train_valid_clouds)  ]
    df_final = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), dfs)
    df_final.columns = ['Train', 'Val', 'Test', 'TrainVal']
    print('SEMILLA=',semilla)
    print('===========')
    print(df_final)

    def cuenta_absoluta(y):
        muestra = collections.Counter(y)
        muestra = pd.DataFrame.from_dict(muestra, orient='index')    
        return muestra

    dfs_cuenta = [cuenta_absoluta(train_clouds),cuenta_absoluta(valid_clouds), cuenta_absoluta(test_clouds), cuenta_absoluta(train_valid_clouds)  ]
    dfs_cuenta_final = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), dfs_cuenta)
    dfs_cuenta_final.columns = ['Train', 'Val', 'Test', 'TrainVal']
    print(dfs_cuenta_final)

    def proporcion_max(y):
        muestra = collections.Counter(y)
        maximo = max(muestra.values())
        for x in muestra:
            muestra[x] = maximo / muestra[x] 
        muestra = pd.DataFrame.from_dict(muestra, orient='index')    
        return muestra

    dfs_max = [proporcion_max(train_clouds),proporcion_max(valid_clouds), proporcion_max(test_clouds), proporcion_max(train_valid_clouds)  ]
    df_final_max = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), dfs_max)
    df_final_max.columns = ['Train', 'Val', 'Test', 'TrainVal']
    print(df_final_max)

    weights = df_final_max.sort_index().TrainVal.values
   
    class_weights = torch.FloatTensor(weights).to(device)
    class_weights = None


le = preprocessing.LabelEncoder()
# RAM180321: He cambiado train_clouds por train_clouds+valid_clouds+test_clouds
model_le = le.fit(train_clouds+valid_clouds+test_clouds)
train_labels_encoded = model_le.transform(train_clouds)
valid_labels_encoded = model_le.transform(valid_clouds)
test_labels_encoded = model_le.transform(test_clouds)
train_valid_encoded = model_le.transform(train_valid_clouds)

# Store label variables as tensors for CNN
train_labels = torch.as_tensor(train_labels_encoded)
valid_labels = torch.as_tensor(valid_labels_encoded)
test_labels = torch.as_tensor(test_labels_encoded)
train_valid_labels = torch.as_tensor(train_valid_encoded)
#print(type(train_labels))

np.all(model_le.classes_ == df_final_max.sort_index().index)



## Let's implement the architecture of the neural network

# To do so, we are going to use the images of the clouds and also the
# the types of cloud to label the image

# The first thing is to generate a dataset that maps each image to the corresponding type
# of cloud
# To do that, we use Dataset class in Pytorch

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    # Pre-trained model expects normalized image
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class mydataset(Dataset):
    def __init__(self, images_labels, images, transform=data_transforms):
        self.images_labels = images_labels
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images_labels)
    def __getitem__(self, idx):
        image = self.images[:,:,:,idx]
        image_label = self.images_labels[idx]
        if self.transform:
            image = self.transform(image)
        # Set images and labels to adequate tensor data types
        # image = image.to(torch.float32)
        # image_label = image_label.to(torch.int64)
        return image, image_label

print("class mydataset definition is OK")

# Generate three dataset calling the class mydataset
trainset = mydataset(train_labels, train_img1)
validset = mydataset(valid_labels, valid_img1)
testset = mydataset(test_labels, test_img1)
trainvalidset = mydataset(train_valid_labels, train_valid_img1)

print("call class mydataset is OK")

# Generate three dataloaders

# RAM290921: Pongo shuffle a False
shuffle_train = True

trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=shuffle_train, num_workers=4)
validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
trainvalidloader = torch.utils.data.DataLoader(trainvalidset, batch_size=12, shuffle=shuffle_train, num_workers=4)

print("call dataloaders are OK")


# Have a look to a minibatch sample

# trainloaderiter = iter(trainloader)
# imageiter, labeliter = trainloaderiter.next()
# print(type(imageiter))
# print(imageiter.shape)
# print(labeliter.shape)

# print("call dataloaders OK")

# Define the arquicture: import ResNet
# It has been adapated to deal with RGB images of 500x500

pretrained = True
# Recordar que el modelo hay que volverlo a definir en: RAM230521:1914
# resnet18 = models.resnet18(pretrained=pretrained)

def model_initialization(pretrained):
    # return models.densenet121(pretrained=pretrained)
    # return models.densenet169(pretrained=pretrained) 
    # return models.alexnet(pretrained=pretrained)
    # return models.vgg16(pretrained=pretrained)
    return models.convnext_tiny(pretrained=pretrained)

resnet18 = model_initialization(pretrained)

# densenet121 = models.densenet121(pretrained=True)

model = resnet18

def train_model(model, trainloader, validloader, testloader, optimizer, criterion, epochs):

    loss_train = []
    loss_valid = []
    loss_test = []
    acc_train = []
    acc_valid = []
    acc_test = []
    label_true = []
    label_pred = []
    pred_list = []
    true_list = []
    print('training on', device)

    for e in range(int(epochs)):
        print('Epoch {}/{} '.format(e, epochs-1))

        model.train()
        epoch_predlist = []
        epoch_truelist = []
        valid_loss =0.
        running_loss = 0.
        testing_loss = 0.
        correct = 0
        total = 0

        for images, labels in trainloader:
            
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        acc_train.append(100*correct/total)
        loss_train.append(running_loss/len(trainloader))

        model.eval()
        with torch.no_grad():

            correct = 0
            total = 0

            for images, labels in validloader:

                images = images.to(device)
                labels = labels.to(device)
                val_out = model(images)
                val_loss = criterion(val_out, labels)
                valid_loss += val_loss.item()
                _, pred = torch.max(val_out, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            acc_valid.append(100*correct/total)
            loss_valid.append(valid_loss/len(validloader))

            correct = 0
            total = 0

            for images, labels in testloader:

                images = images.to(device)
                labels = labels.to(device)
                test_out = model(images)
                test_loss = criterion(test_out, labels)
                testing_loss += test_loss.item()
                _, pred = torch.max(test_out , 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                epoch_predlist.extend(pred.tolist())
                epoch_truelist.extend(labels.tolist())

            pred_list.append(epoch_predlist)
            true_list.append(epoch_truelist)
            acc_test.append(100*correct/total)
            loss_test.append(testing_loss/len(testloader))

    # generate csv file with the losses and the accuracy
    dict_plot = {'train loss': loss_train, 'valid loss': loss_valid, 'test loss': loss_test, 'acc train':acc_train, 'acc valid': acc_valid, 'acc test': acc_test}
    pd_plot = pd.DataFrame(dict_plot)
    pd_plot.to_csv('./dataplot_resnet18.csv')

    # generate the csv file with the predictions and true test_labels
    pd_pred = pd.DataFrame(pred_list)
    pd_pred.to_csv('./pred_resnet18.csv')
    pd_true = pd.DataFrame(true_list)
    pd_true.to_csv('./true_resnet18.csv')

    return model



startime = timeit.default_timer()

# Change the last layer of the model to adapt it to our problem with 9 classes
# num_classes = 9
num_classes = len(set(train_valid_clouds + test_clouds))

# myfc = nn.Sequential(
#     nn.Dropout(0.25),
#     nn.Linear(in_features=512, out_features=num_classes)
# )

myfc = nn.Linear(in_features=512, out_features=num_classes)

model.fc = myfc
model = model.to(device)

# Define optimizer and criterion
from torch import optim
#optim = optim.Adagrad(model.parameters(), lr=0.001)

learning_rate = 0.00001
# learning_rate = 0.000001
# weight_decay = 0.0001
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
#optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if(class_weights is not None):
    criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
    criterion = nn.CrossEntropyLoss()

# criterion = nn.CrossEntropyLoss()

# RAM190321: Reproducibility
# RAM280222: Hasta hoy estaba con semilla=10, le pongo 0
random.seed(semilla)
torch.manual_seed(semilla)

# Call the function to train and evaluate the model
# model = train_model(model, trainloader, validloader, testloader, optimizer, criterion, epochs=4000)
model = train_model(model, trainloader, validloader, testloader, optimizer, criterion, epochs=10000)
# model = train_model(model, trainloader, validloader, testloader, optim, criterion, epochs=50)

import shutil
shutil.copy('dataplot_resnet18.csv',  'dataplot_resnet18_orig.csv')
shutil.copy('pred_resnet18.csv', 'pred_resnet18_orig.csv')
shutil.copy('true_resnet18.csv', 'true_resnet18_orig.csv')

torch.save(model.state_dict(), 'checkpoint_resnet18_orig.pth')

# RAM230521:1914 Esto es para volver a llamar a la red con todos los datos
if(True):
    data = pd.read_csv('dataplot_resnet18.csv', sep=',')
    epochs_star = data.index[data['valid loss']==data['valid loss'].min()].values[0]
    # resnet18 = models.resnet18(pretrained=pretrained)
    # models.resnet34(pretrained=pretrained)

    resnet18 = model_initialization(pretrained)
    
    
    model = resnet18
    myfc = nn.Linear(in_features=512, out_features=num_classes)
    # myfc = nn.Sequential(
    #    nn.Dropout(0.25),
    #     nn.Linear(in_features=512, out_features=num_classes)
    # )
    model.fc = myfc
    model = model.to(device)
    # RAM190321: Reproducibility
    # RAM280222: Hasta hoy estaba con 10, le pongo 0
    random.seed(semilla)
    torch.manual_seed(semilla)
    from torch import optim
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    model = train_model(model, trainvalidloader, validloader, testloader, optimizer, criterion, epochs=epochs_star+1)


torch.save(model.state_dict(), 'checkpoint_resnet18.pth')
# state_dict = torch.load('checkpoint_resnet18.pth')
# model.load_state_dict(state_dict)


stoptime = timeit.default_timer()
print('Execution time:', stoptime - startime)

# RAM210521:1929 A partir de aquí, mejor usar el fichero dataplot_RN18.ipynb
# Obtain the labels for the confusion matrix

if(False):
    test_clouds_confusion = pd.Series(test_clouds)
    #test_clouds_confusion = test_clouds_confusion['types']
    print(test_clouds_confusion.value_counts())
    test_clouds_confusion = test_clouds_confusion.tolist()
    print('classes_:', list(model_le.classes_))
    test_clouds_confusion = model_le.transform(test_clouds_confusion)
    test_clouds_confusion = pd.Series(test_clouds_confusion)
    print(test_clouds_confusion.value_counts())

#####################################################################
# RAM180321 Esto lo he copiado del fichero de RF_v1 de Victor
#####################################################################

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
import timeit
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import accuracy_score

heinlered_train = train_heinlered.drop(labels=['datetimes'], axis=1)
X_train = heinlered_train.iloc[:,:-1]
y_train = heinlered_train.iloc[:,-1]

heinlered_valid = valid_heinlered.drop(labels=['datetimes'], axis=1)
X_valid = heinlered_valid.iloc[:,:-1]
y_valid = heinlered_valid.iloc[:,-1]

heinlered_test = test_heinlered.drop(labels=['datetimes'], axis=1)
X_test = heinlered_test.iloc[:,:-1]
y_test = heinlered_test.iloc[:,-1]

# We generate a training plus validation partition
# to be used in the hyper-parameter turning during the cv

# Estos porcentajes que vienen ya no deben de ser así
# This partition accounts for 80% of the total data,
# maintaining the same percentages of types of clouds

X_train_valid = np.concatenate((X_train, X_valid))
y_train_valid = np.concatenate((y_train, y_valid))


##### Hyper-parameter tuning using RandomizedSearchCV

# Empty random forest classifier

clf = RandomForestClassifier()

## Param grid

# Criterion

criterion = ['gini','entropy']

# Number of trees
n_estimators = [int(x) for x in np.linspace(40, 800, 20)]

# Number of features to consider at every split
max_features = ['auto', 'log2']

# Minimum number of samples required to split an internal node
min_samples_split = sp_uniform(0.0, 1.0)

param_grid = {
  'criterion':criterion,
  'n_estimators':n_estimators,
  'max_features':max_features,
  'min_samples_split':min_samples_split}

# Number of iterations
budget = 200


rs_split = PredefinedSplit(test_fold=[-1]*len(y_train)+[0]*len(y_valid))

# Generate RandomizedSearchCV object
# cv = 4 is used to keep the similarity of 60% training and 20 validation in CNN
model_tune_rf = RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=budget,
                                   refit=True,
                                   scoring='accuracy',
                                   cv=rs_split, 
                                   # verbose=1,
                                   random_state = 10, 
                                   n_jobs=-1)


# RAM190321: Reproducibility
# RAM280222: Hasta hoy estaba con 10, le pongo 0
random.seed(semilla)

# Fit the model
model_tune_rf = model_tune_rf.fit(X_train_valid, y_train_valid)

# Predict with the best found parameters
predict = model_tune_rf.predict(X_test)

# Print the best hyper-parameters
print("Best hyper-parameter setting :{}" .format(model_tune_rf.best_params_))

'''
Best hyper-parameter setting :{'criterion': 'entropy', 'max_features': 'auto',
'min_samples_split': 0.005156348033917069, 'n_estimators': 320}
'''
# Print the best accuracy score:
print("Best validation accuracy :{}" .format(model_tune_rf.best_score_))
'''
Best accuracy :0.8025111521304414
'''
# Print the best accuracy
print("Best test accuracy :{}" .format(accuracy_score(y_test.values, predict)))
'''
Best accuracy :0.7929292929292929
'''

# Obtain the csv files to export
pd_true1 = pd.DataFrame(y_test)
pd_true1.to_csv('./truev1.csv')

pd_pred1 = pd.DataFrame(predict)
pd_pred1.to_csv('./predv1.csv')

