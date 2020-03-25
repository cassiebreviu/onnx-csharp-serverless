# ONNX: No, it's not a Pokemon! Deploy your ONNX model with C# and Azure Functions

Ok you got a ML model working in Jupyter notebooks, now what? Lets deploy it! There are many ways to opperationalize your model. In this tutorial we are going to be using a model created with Python SciKit Learn from [this blog post]() to classify wine based on the description from a wine magazine. We are going to take that model, update it to use a [pipeline]() and export it to an ONNX format. The reason we want to use ONNX format is because this is what will allow us to deploy it to many different platforms. There are many other benefits (such as performance) to using onnx. Learn more about that [here](). Since I ♥ C# we are going to use with the [onnxruntime nuget library]() available for dotnet. However, if you prefer to use a different language many are supported. You can find all supported languages [here]().

### Prerquesites

- [AML Azure Resource]() with a Notebook VM instance created
  - If you prefer to create the model locally I recommend downloading [Anaconda](). However, this tutorial is written as if you are using AML.
- [VS Code]() with [Azure Functions extension]()
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
""""""""''''''''
