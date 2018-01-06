from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import InceptionV3
from keras.applications import Xception
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg16_pi
from keras.applications.vgg19 import preprocess_input as vgg19_pi
from keras.applications.inception_v3 import preprocess_input as iv3_pi
from keras.applications.xception import preprocess_input as xc_pi
from skimage.io import imread
import numpy as np
import os
import glob
from keras.models import Model

def extract(input_folder, feature_name, feature_folder):
    X = []
    if feature_name == "vgg19":
        base_model = VGG19(weights='imagenet')
        save_file_name = 'vgg19_features'

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

        for filename in glob.glob(os.path.join(input_folder,'*.jpg')):

            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = vgg19_pi(x)
            features = model.predict(x)
            X.append(features)

    elif feature_name == "vgg16":
        base_model = VGG16(weights='imagenet')
        save_file_name = 'vgg16_features'

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):

            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = vgg16_pi(x)
            features = model.predict(x)
            X.append(features)


    elif feature_name == 'inception':
        base_model = InceptionV3(weights='imagenet')
        save_file_name = 'inceptionV3_features'

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
            print(filename)

            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = iv3_pi(x)
            features = model.predict(x)
            X.append(features)


    elif feature_name == 'xception':
        base_model = Xception(weights='imagenet')
        save_file_name = 'xception_features'

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = xc_pi(x)
            features = model.predict(x)
            X.append(features)

    elif feature_name == 'resnet50':
        base_model = ResNet50(weights='imagenet')
        save_file_name = 'resnet50_features'

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            avg_pool_feature = model.predict(x)
            X.append(avg_pool_feature)

    elif feature_name == 'lbp':
        save_file_name = 'lbp_features'

        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
            filename = imread(filename,as_grey=True)
            lbt_image = local_binary_pattern(filename, P=24, R=3, method='uniform')
            (lbt_hist, _) = np.histogram(lbt_image.ravel(), bins=int(lbt_image.max() + 1), range=(0, 24 + 2))
            X.append(lbt_hist)

    elif feature_name == 'hog':
        save_file_name = 'hog_features'
        for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
            filename = imread(filename, as_grey=True)
            lbt_image = hog(filename, block_norm='L2')
            X.append(lbt_image)

    else:
        print('Not support model ' + feature_name)
    if X:
        X = np.array(X)
        X = np.squeeze(X)
        save_file = os.path.join(feature_folder, save_file_name + '.npy')
        #if feature_name == 'lbp' or feature_name == 'hog':
        X = X.T
        if not os.path.exists(feature_folder):
            os.makedirs(feature_folder)
        else:
            if os.path.isfile(save_file):
                os.remove(save_file)
        np.save(save_file,X)
    else:
        print('No input received')
    return X

extract('image', 'lbp', 'Feature')
extract('image', 'hog', 'Feature')
extract('image', 'vgg16', 'Feature')
extract('image', 'vgg19', 'Feature')
extract('image', 'inception', 'Feature')
extract('image', 'xception', 'Feature')
extract('image', 'resnet50', 'Feature')
