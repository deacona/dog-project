import os
from flask import Flask
from flask import render_template, request
import cv2
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential


#######################################################################


# instantiate web app
app = Flask(__name__)

# cleaned list of dog names for prediction output
dog_names = [
    "Affenpinscher",
    "Afghan Hound",
    "Airedale Terrier",
    "Akita",
    "Alaskan Malamute",
    "American Eskimo Dog",
    "American Foxhound",
    "American Staffordshire Terrier",
    "American Water Spaniel",
    "Anatolian Shepherd Dog",
    "Australian Cattle Dog",
    "Australian Shepherd",
    "Australian Terrier",
    "Basenji",
    "Basset Hound",
    "Beagle",
    "Bearded Collie",
    "Beauceron",
    "Bedlington Terrier",
    "Belgian Malinois",
    "Belgian Sheepdog",
    "Belgian Tervuren",
    "Bernese Mountain Dog",
    "Bichon Frise",
    "Black And Tan Coonhound",
    "Black Russian Terrier",
    "Bloodhound",
    "Bluetick Coonhound",
    "Border Collie",
    "Border Terrier",
    "Borzoi",
    "Boston Terrier",
    "Bouvier Des Flandres",
    "Boxer",
    "Boykin Spaniel",
    "Briard",
    "Brittany",
    "Brussels Griffon",
    "Bull Terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn Terrier",
    "Canaan Dog",
    "Cane Corso",
    "Cardigan Welsh Corgi",
    "Cavalier King Charles Spaniel",
    "Chesapeake Bay Retriever",
    "Chihuahua",
    "Chinese Crested",
    "Chinese Shar-Pei",
    "Chow Chow",
    "Clumber Spaniel",
    "Cocker Spaniel",
    "Collie",
    "Curly-Coated Retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie Dinmont Terrier",
    "Doberman Pinscher",
    "Dogue De Bordeaux",
    "English Cocker Spaniel",
    "English Setter",
    "English Springer Spaniel",
    "English Toy Spaniel",
    "Entlebucher Mountain Dog",
    "Field Spaniel",
    "Finnish Spitz",
    "Flat-Coated Retriever",
    "French Bulldog",
    "German Pinscher",
    "German Shepherd Dog",
    "German Shorthaired Pointer",
    "German Wirehaired Pointer",
    "Giant Schnauzer",
    "Glen Of Imaal Terrier",
    "Golden Retriever",
    "Gordon Setter",
    "Great Dane",
    "Great Pyrenees",
    "Greater Swiss Mountain Dog",
    "Greyhound",
    "Havanese",
    "Ibizan Hound",
    "Icelandic Sheepdog",
    "Irish Red And White Setter",
    "Irish Setter",
    "Irish Terrier",
    "Irish Water Spaniel",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Japanese Chin",
    "Keeshond",
    "Kerry Blue Terrier",
    "Komondor",
    "Kuvasz",
    "Labrador Retriever",
    "Lakeland Terrier",
    "Leonberger",
    "Lhasa Apso",
    "Lowchen",
    "Maltese",
    "Manchester Terrier",
    "Mastiff",
    "Miniature Schnauzer",
    "Neapolitan Mastiff",
    "Newfoundland",
    "Norfolk Terrier",
    "Norwegian Buhund",
    "Norwegian Elkhound",
    "Norwegian Lundehund",
    "Norwich Terrier",
    "Nova Scotia Duck Tolling Retriever",
    "Old English Sheepdog",
    "Otterhound",
    "Papillon",
    "Parson Russell Terrier",
    "Pekingese",
    "Pembroke Welsh Corgi",
    "Petit Basset Griffon Vendeen",
    "Pharaoh Hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese Water Dog",
    "Saint Bernard",
    "Silky Terrier",
    "Smooth Fox Terrier",
    "Tibetan Mastiff",
    "Welsh Springer Spaniel",
    "Wirehaired Pointing Griffon",
    "Xoloitzcuintli",
    "Yorkshire Terrier",
]


#######################################################################


def face_detector(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (boolean) - True if face(s) detected, False if not
    """
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        "../haarcascades/haarcascade_frontalface_alt.xml"
    )

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (4D tensor) - 4D array of shape (1, 224, 224, 3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def dog_detector(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (boolean) - True if predicted category key falls within range of dog keys (151-268 inclusive), False otherwise
    """

    # define ResNet50 model
    ResNet50_model = ResNet50(weights="imagenet")
    # prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return (prediction <= 268) & (prediction >= 151)


def predict_breed(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        prediction - Dog breed predicted by the model
    """

    # use saved model
    # saved_model = load_model("../saved_models/model.final.hdf5")
    # having to do it this way as getting errors when using load_model
    saved_model = Sequential()
    saved_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    saved_model.add(Dense(133, activation="softmax"))
    saved_model.load_weights("../saved_models/weights.best.ResNet50.hdf5")

    # extract bottleneck features
    tensor = path_to_tensor(img_path)
    bottleneck_feature = ResNet50(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )

    # return dog breed that is predicted by the model
    predicted_vector = saved_model.predict(bottleneck_feature)
    prediction = dog_names[np.argmax(predicted_vector)]
    return prediction


def what_am_i(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        title - Info on species and breed (as appropriate)
    """

    if img_path == "static/":
        return None

    species = "Other"
    if dog_detector(img_path):
        species = "Dog"
    elif face_detector(img_path):
        species = "Human"

    if species == "Other":
        title = "You are not a Human or a Dog!"
    else:
        breed = predict_breed(img_path)

        ## not perfect but will do for now
        if breed[0] in "AEIOU":
            indef_article = "an"
        else:
            indef_article = "a"

        title = "You are a {0}, you look like {1} {2}".format(
            species, indef_article, breed
        )

    return title


#######################################################################


@app.route("/")
@app.route("/index")
def index():
    """
    Parse image paths, make predictions and display web page
    """
    images = os.listdir("static")
    selection = request.args.get("selection", "")
    # print("static/"+selection)
    prediction = what_am_i("static/" + selection)
    return render_template(
        "master.html", images=images, selection=selection, prediction=prediction
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
