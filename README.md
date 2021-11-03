# MetaCheX

## Dataset Setup
### ChestX-ray14 ([link](https://www.kaggle.com/nih-chest-xrays/data))
- Download from Kaggle

```
mkdir data && cd data
kaggle datasets download -d nih-chest-xrays/data
```

- Put all the images into `$MCX_ROOT/data/nih/images`

```
mkdir -p nih/images
unzip data.zip -d nih/images
rm *.pdf && rm *.txt && rm BBox_List_2017.csv
mv */* .
rmdir *
```

### COVID-19 RADIOGRAPHY ([link](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database))
- Download from Kaggle
```
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

- Put all images into `$MCX_ROOT/data/COVID-19_Radiography_Dataset/images`
```
mkdir -p COVID-19_Radiography_Dataset/images
unzip covid19-radiography-database.zip -d COVID-19_Radiography_Dataset/images
rm *.xlsx
mv */* .
rmdir *
```

### covid-chestxray-dataset ([link](https://github.com/ieee8023/covid-chestxray-dataset.git))
- Clone GitHub repo in data folder

```
git clone git@github.com:ieee8023/covid-chestxray-dataset.git
```
