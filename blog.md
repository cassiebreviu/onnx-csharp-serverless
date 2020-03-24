# ONNX: No, it's not a Pokemon! Deploy your ONNX model with C# and Azure Functions

Ok you got a ML model working in Jupyter notebooks, now what? Lets deploy it! There are many ways to opperationalize your model. In this tutorial we are going to be using a model created with Python SciKit Learn from [this blog post]() to classify wine based on the description from a wine magazine. We are going to take that model, update it to use a [pipeline]() and export it to an ONNX format. The reason we want to use ONNX format is because this is what will allow us to deploy it to many different platforms. There are many other benefits (such as performance) to using onnx. Learn more about that [here](). Since I ♥ C# we are going to use with the [onnxruntime nuget library]() available for dotnet. However, if you prefer to use a different language many are supported. You can find all supported languages [here]().

### Prerquesites

- [AML Azure Resource]() with a Notebook VM instance created
  - If you prefer to create the model locally I recommend downloading [Anaconda]()
- [VS Code]() with [Azure Functions extension]()
- [.NET core 3.1](https://dotnet.microsoft.com/download)

### What is Open Neural Network Exchange (ONNX)?

Thanks for asking! ONNX is an open/common file format to enable you to use models with a variety of frameworks, tools, runtimes, and compilers. Once the model is exported to the ONNX format then use the ONNX Runtime a cross-platform, high performance scoring engine for ML models.

Build models in the Tensorflow, Keras, PyTorch, scikit-learn, CoreML, and other popular supported formats can be converted to the standard ONNX format, providing framework interoperability and helping to maximize the reach of hardware optimization.

### Why you should use it?

You are full of great questions. The answer is simple: it gives you the ability to use the same model and application code across different platforms. This means I can create this model in Python with SciKit Learn and use the resulting model in C#! Say whaaat? Yes, that is right. Save it to ONNX format then run it in csharp with the onnxruntime!

## Create the Model

I have set this up to give you **two options**. The _first option_ is to use the model from the previous blog post to classify wine. The _second option_ is to skip to update your existing code to export the model in ONNX format.

### Option 1: To Use the Wine Model Code Provided

- Clone Jupyter Notebook to Create the Model

```shell
git clone REPO HERE
```

- Run the code to get the model
     <!--TODO finish writing steps here-->
  > [!NOTE]
  > There are different package depending on what library you used to create the model. Since we use scikit learn we will use the skl2onnx package to export our trained model

-pip install skl2onnx
-restart kernel
-import packages

```# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
```

### Option 2: To Use Your Own Model

To use your own model there are a few things to consider. You will need to update the export code based on your model. I will go over a few different ways to export the model. Hopefully your model will fall into one of the categories provided.

<!--TODO finish writing steps here, include samples for different types of models to export to onnx-->

## Save Model to Azure Storage

Now that we have exported the model into onnx format lets save it to [Azure Storage]().

- Go to [Azure Portal](https://portal.azure.com)

## Deploy Model with Azure Functions

### Create Azure Function

- crtl shift p
- search create function
- select create and ok to the folder
  -ADD REST OF STEPS HERE

- hit f5 to run project and test that its working
- if prompted to install \azure-functions-core-tools
  > node lib/install.js click install

### Install the nuget package

onnx runtime

```dotnetcli
dotnet add package Microsoft.ML.OnnxRuntime --version 1.2.0
```

tensor numerics

```dotnetcli
dotnet add package System.Numerics.Tensors --version 0.1.0
```

add the onnx runtime and tensor

```csharp
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
```

# Resources

[ONNX Docs](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx)

[Onnx CSharp API Docs](https://github.com/microsoft/onnxruntime/blob/master/docs/CSharp_API.md)
[Scikit learn pipeline onnx](http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html#l-example-tfidfvectorizer)
[](https://github.com/Microsoft/onnxruntime)
