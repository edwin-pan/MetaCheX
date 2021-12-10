# Paths
PATH_TO_DATA_FOLDER = 'data'
#'./data'
NIH_IMAGES = 'nih/images'
NIH_METADATA_PATH = 'nih/Data_Entry_2017.csv'
COVID_19_RADIOGRAPHY_IMAGES = 'COVID-19_Radiography_Dataset/images' ## note labels are in the filenames
COVID_CHESTXRAY_IMAGES = 'covid-chestxray-dataset/images'
COVID_CHESTXRAY_METADATA_PATH = 'covid-chestxray-dataset/metadata.csv'

# Constants
IMAGE_SIZE = 224
SAMPLE_MIN = 5

# Tsne plot classes
TSNE_PARENT_CLASSES = ['COVID-19', 'Pneumonia']
TSNE_CHILD_CLASSES = ['COVID-19|Pneumonia']
TSNE_DISTANCE_CLASSES = ['Pneumonia', 'Hernia', 'Lung_Opacity']