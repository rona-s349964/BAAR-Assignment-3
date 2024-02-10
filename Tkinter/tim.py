# Import the required modules
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch                     # PyTorch library for AI models
import torchvision               # PyTorch library for computer vision

class ImageClassifierApp(tk.Tk): # Inheritance
    """
    Subclass of  tk.Tk; inherits methods and attributes of 
    Tkinter class.
    """

    def __init__(self):
        """
        Constructor method that initializes an instance of the ImageClassifierApp class.
        Sets up the main window with a title and dimensions.
        Creates sub-frames within the main window to hold widgets.
        Defines methods for handling button clicks and image loading events.
        """

        super().__init__()             # Method overriding

        # Initialize tkinter window
        self.title("Image Classifier")
        self.geometry("600x400")

        # Creating a frame for organising widgets in it
        self.frame = ttk.Frame(self)
        self.frame.pack(padx=10, pady=10)

        # Create and configure all the widgets
        self.label = ttk.Label(self.frame, text="Enter the path to an image file and click Classify")
        self.label.grid(row=0, column=0, columnspan=2)
        self.url = tk.StringVar()
        self.entry = ttk.Entry(self.frame, textvariable=self.url, width=80)
        self.entry.grid(row=1, column=0, padx=5, pady=5)
        self.entry.focus()
        self.button = ttk.Button(self.frame, text="Classify", command=self.classify)
        self.button.grid(row=1, column=1, padx=5, pady=5)
        self.image_label = ttk.Label(self.frame)
        self.image_label.grid(row=2, column=0, columnspan=2)
        self.result_label = ttk.Label(self.frame, text="")
        self.result_label.grid(row=3, column=0, columnspan=2)
        self.status_label = ttk.Label(self.frame, text="")
        self.status_label.grid(row=4, column=0, columnspan=2)

        # Creation of AI model using PyTorch
        # Polymorphism: The model can be any PyTorch model (in this case, ResNet18)
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()

        # Reading class names for display of results
        self.class_names = []
        with open("imagenet_classes.txt") as f:
            for line in f:
                self.class_names.append(line.strip())

    def classify(self):
        """
        Classifies the image and displays the results utilizing the 
        AI model.
        """

        # Method overriding: Overrides the generic 'classify' method

        # Getting URL of image
        # this should be an absolute path of the image in the system
        url = self.url.get()

        try:
            # Open and pre-process image using PIL and torchvision
            image = Image.open(url)
            image = image.resize((224, 224))
            image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk

            image_tensor = torchvision.transforms.ToTensor()(image)
            image_tensor = image_tensor.unsqueeze(0)

            # PErform classification using the AI model
            output = self.model(image_tensor)
            index = torch.argmax(output)
            class_name = self.class_names[index]

            # Finally, we update the result label with the classifier results
            self.status_label.config(text="Classification done successfully.") 
            self.result_label.config(text="Classifier result: " + class_name)
        except:
            self.result_label.config(text="")
            self.status_label.config(text="Classification failed. Please enter a valid image path.")


# Create an instance of the image classifier and start it
app = ImageClassifierApp()
app.mainloop()
