# Import libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetMultiLabel, self).__init__()
        self.name = "densenetclassifier"
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        # Attaching our classifier to the pretrained model
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet(x)
        x = self.sigmoid(x)
        return x

def find_matching_recipes(identified_items, ingredient_and_instructions_csv, max_recipes_per_item=5):
    # Load the CSV file containing ingredient and recipe information
    df = pd.read_csv(ingredient_and_instructions_csv)

    # Count occurrences of each identified class in the Ingredients column
    class_counts = {}
    for item in identified_items:
        class_counts[item] = df['Ingredients'].str.contains(item, case=False).sum()

    # Sort the identified classes based on their occurrence count (fruits/vegetables first)
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)

    matching_recipes = {}
    for item in sorted_classes:
        # Find matching recipes for the current identified item
        matching_rows = df[df['Ingredients'].str.contains(item, case=False)]
        matching_rows = matching_rows.head(max_recipes_per_item)

        # Extract the corresponding Title and Cleaned_Ingredients columns
        titles = matching_rows['Title'].tolist()
        image_names = matching_rows['Image_Name'].tolist()
        ingredients = matching_rows['Cleaned_Ingredients'].tolist()
        instructions = matching_rows['Instructions'].tolist()

        # Store the matching recipes and their ingredients in the dictionary
        for title, image_name, ingredients, instructions in zip(titles, image_names, ingredients, instructions):
            if item in matching_recipes:
                matching_recipes[item].append((title, image_name, ingredients, instructions))
            else:
                matching_recipes[item] = [(title, image_name, ingredients, instructions)]

    return matching_recipes


def display_image(image_path):
    root = tk.Tk()
    root.title("Image Viewer")

    img = Image.open(image_path)
    img = img.resize((274, 169))
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=img_tk)
    label.pack()

    root.mainloop()

def recipe_scout(predicted_classes):
  print("Recipe Scout has predicted the following classes:", predicted_classes, "\n")

  # Taking in user input for the number of recipes they want per class
  num_recipes = int(input("How many recipe recommendations would you like per ingredient? (Max of 10): "))

  while (num_recipes > 10 or num_recipes <= 0):
    if num_recipes <= 10 and num_recipes > 0:
      print("Received. See recipes below.")
    else:
      num_recipes = int(input("Invalid input, please try again: "))

  # From the recipe dataset, pull and save the user-requested number of recipes per class
  matching_recipes = find_matching_recipes(predicted_classes, r"C:\Users\ricky\PycharmProjects\aps360\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", num_recipes)

  # Display the names of each recipe and a corresponding image
  for i in range(len(predicted_classes)):
    print("\u0332".join("Recipes with " + str(predicted_classes[i]) + ":"))
    for title in range(num_recipes):
      print(str(title+1) + ".", matching_recipes[predicted_classes[i]][title][0])
      image_name = matching_recipes[predicted_classes[i]][title][1] + '.jpg'
      image_path = r"C:\Users\ricky\PycharmProjects\aps360\Food Images\Food Images" + "\\" + image_name
      display_image(image_path)

    print("\n")

  print("\n")

  print("\u0332".join("To read the full list of ingredients & instructions for a recipe, follow these instructions: "))

  while (True):
    # Taking user input for which recipe to display ingredients/instructions for
    read_more_class = str(input("Enter the class (ex. potato), or 'stop' to stop: "))
    if read_more_class.lower() == 'stop':
      break
    read_more_num = int(input("Enter the corresponding number (ex: 1), or 0 to stop: "))
    if read_more_num == 0:
      break
    print("\n")
    print("\u0332".join("List of Ingredients: "))
    # Converting ingredient string to list and printing each item on a separate line
    ingredients = eval(matching_recipes[read_more_class][read_more_num-1][2])
    print(*ingredients, sep = '\n')
    print("\n")
    print("\u0332".join("List of Instructions"))
    # Splitting instructions by the '\n' tag and printing on separate lines
    instructions = matching_recipes[read_more_class][read_more_num-1][3].split('\n')
    i = 1
    for sentence in instructions:
      print(str(i) + ". " + sentence)
      i += 1

    # Loops until user chooses to exit using 'stop' or '0'
    print("\u0332".join("\nWant to try another recipe?"))

  print("Thank you for using Recipe Scout. Hope to see you again soon!")

use_cuda = False
densenetclassifier = DenseNetMultiLabel(20)

if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    densenetclassifier.cuda()
else:
    device = torch.device("cpu")

# Load in the model checkpoint
state = torch.load('model_densenetclassifier_bs150_lr0.001_dr0.9_thresh0.5_epoch9', map_location=device)
state = {k.replace('module.', ''): v for k, v in state.items()}
densenetclassifier.load_state_dict(state)
densenetclassifier.eval()

# Load in a single image to perform the prediction on
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image_path = str(input("Enter complete image path here: ")).strip('\"')

image = Image.open(image_path).convert('RGB')
input_img = transform(image).unsqueeze(0)

# Set prediction threshold
threshold = 0.5

# Make the prediction
output = densenetclassifier(input_img)

# Convert the tensor to a NumPy array for formatting
output_np = output.cpu().detach().numpy()

# Define the 20 possible classes
classes = ['potato', 'tomato', 'carrot', 'onion', 'cabbage', 'broccoli', 'bell pepper', 'spinach', 'cauliflower',
               'green bean', 'apple', 'banana', 'orange', 'strawberry', 'grape', 'watermelon', 'pineapple',
               'mango', 'peach', 'pear']

for i, probabilities in enumerate(output_np):
    print("The following is the model's confidence on each of fruits and vegetables present in the photo: ")
    for j, probability in enumerate(probabilities):
        print("  {}: {:.2%}".format(classes[j], probability))

output = output.detach().cpu()
best = (output >= threshold).int()

# Interpret the tensor and save the predicted classes
predicted_indices = best.squeeze().nonzero().flatten().tolist()
predicted_classes = [classes[i] for i in predicted_indices]

recipe_scout(predicted_classes)

