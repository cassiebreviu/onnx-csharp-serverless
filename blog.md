# ONNX: No, it's not a Pokemon! Deploy your ONNX model with C# and Azure Functions

Ok you got a ML model working in Jupyter notebooks, now what? Lets deploy it! There are many ways to opperationalize your model. In this tutorial we are going to be using a model created with Python SciKit Learn from [this blog post]() to classify wine based on the description from a wine magazine. We are going to take that model, update it to use a [pipeline]() and export it to an ONNX format. The reason we want to use ONNX format is because this is what will allow us to deploy it to many different platforms. There are many other benefits (such as performance) to using onnx. Learn more about that [here](). Since I ♥ C# we are going to use with the [onnxruntime nuget library]() available for dotnet. However, if you prefer to use a different language many are supported. You can find all supported languages [here]().

### Prerquesites

- [Create a Free Azure Account!]()
- [AML Azure Resource]() with a Notebook VM instance created
  - If you prefer to create the model locally I recommend downloading [Anaconda](). However, this tutorial is written as if you are using AML.
- [VS Code]()
- [.NET core 3.1](https://dotnet.microsoft.com/download)

### What is Open Neural Network Exchange (ONNX)?

Thanks for asking! ONNX is an open/common file format to enable you to use models with a variety of frameworks, tools, runtimes, and compilers. Once the model is exported to the ONNX format then use the ONNX Runtime a cross-platform, high performance scoring engine for ML models.

Build models in the Tensorflow, Keras, PyTorch, scikit-learn, CoreML, and other popular supported formats can be converted to the standard ONNX format, providing framework interoperability and helping to maximize the reach of hardware optimization.

### Why should you use it?

You are full of great questions. The answer is simple: it gives you the ability to use the same model and application code across different platforms. This means I can create this model in Python with SciKit Learn and use the resulting model in C#! Say whaaat? Yes, that is right. Save it to ONNX format then run it in csharp with the onnxruntime!

## Create the Model

I have a model from the previous blog post to classify wine quality that we will use as the example model. See the note below if you have your own model you would like to use. Additionaly, you should already have an Azure Machine Learning (AML) Studio created with a Compute resource. If not, follow [these steps]() to create one.

> [!NOTE]
> To use your own model visit the [ONNX Github tutorials](https://github.com/onnx/tutorials#converting-to-onnx-format) to see how to convert different frameworks and tools.

#### 1. Get Notebook

- Open JupyterLab for the compute instance in AML
- Click the Terminal to open a terminal tab
- Clone the github repo

```shell
git clone https://github.com/cassieview/onnx-csharp-serverless.git
```

- The `onnx-csharp-serverless` folder will appear. Navigate to the Jupyter Notebook `onnx-csharp-serverless/notebook/wine-nlp-onnx.ipynb`.

#### 2. Install the package

ONNX has different packages for python conversion to the ONNX format. Since we used SciKit Learn we will use the `skl2onnx` package to export our trained model

- In the terminal install the package with the following command

```python
pip install skl2onnx
```

#### 3. Run the code

The NLP notebook provided goes over how to create a basic bag-of-words style NLP model. Run each cell until you get to the `Export the Model` step near the bottom of the notebook.

The first cell in the export block of the notebook is importing the ONNX packages.

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
```

The next cell is running `convert_sklearn` and sending in the the following values:

- The first parameter is a trained classifier or pipeline. In our example we are using a pipeline with 2 steps to vectorize and classify the text.
- The second parameter is a string name value of your choice.
- Lastly setting the `initial_types` This is a list with each list item a tuple of a variable name and a type.

```python
model_onnx = convert_sklearn(pipeline,
                             "quality",
                             initial_types=[("input", StringTensorType([None, 1]))])
```

Lastly we will simply export the model.

```python
with open("pipeline_quality.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())
```

#### 4. Save Model to Azure Storage

Now that we have exported the model into ONNX format lets save it to Azure Storage.

- Follow [these steps](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal?WT_cassieb) to create a storage account and upload the model created.

## Deploy Model with Azure Functions

#### 1. Create Azure Function

If you don't already have the Azure Function extension, follow the below steps to install it:

- Install the Azure Functions extension. You can use the Azure Functions extension to create and test functions and deploy them to Azure.
- In Visual Studio Code, open Extensions and search for azure functions, or select this [link](vscode:extension/ms-azuretools.vscode-azurefunctions)
- Select Install to install the extension for Visual Studio Code:

Once its installed we can now use VS Code to create our function using the command pallet.

- Hit `CTRL-SHIFT-P` to openthe command pallet
- Type `create function` and select the create function option
- From the popup select `Create new project` and create a folder for the project
- When prompted for a language select C#. Note that you have many language options with functions!
- Next select a template. We want an HttpTrigger and give it a name
- Next it will prompt you for a namespace. I used Wine.Function but feel free to name it as you wish.
- Access Rights are next, select `Function`
- Select `open in current window`
- You should be prompted in the bottom right corner to restore packages. If not you can always open the terminal and run `dotnet restore` to restore nuget packages.
- Hit `F5` to run project and test that its working
- Once the function is up there will be a localhost endpoint displayed in the terminal output of VS Code. Paste that into a browser with a query string to test that it is working. The endpoint will look something like this `http://localhost:7071/api/Something?name=test`
- The result in the browser should look something like this `Hello, test. This HTTP triggered function executed successfully.
- Stop the run.

#### 2. Install the Nuget ONNX Packages

Sweet, we now have an Azure Function ready to go. Lets install the nuget package we need to inference with our exported model in C#.

Open the terminal and run the below commands

```dotnetcli
dotnet add package Microsoft.ML.OnnxRuntime --version 1.2.0
dotnet add package System.Numerics.Tensors --version 0.1.0
```

Import the libraries at the top of the csharp class.

```csharp
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
```

#### 3. Update the Code

Copy and paste the below code into the class you created:

```csharp

```

Update the Storage Account connection parameter to your storage account.

#### 4. Test the endpoint

#### 5. Deploy the Endpoint to Azure

# Resources

[ONNX Docs](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx)

[Onnx CSharp API Docs](https://github.com/microsoft/onnxruntime/blob/master/docs/CSharp_API.md)
[Scikit learn pipeline onnx](http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html#l-example-tfidfvectorizer)
[](https://github.com/Microsoft/onnxruntime)
