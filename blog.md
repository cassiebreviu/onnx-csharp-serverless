# ONNX: No, it's not a Pokemon! Deploy your ONNX model with C# and Azure Functions

Ok you got a ML model working in Jupyter notebooks, now what? Lets deploy it! There are many ways to opperationalize your model. In this tutorial we are going to be using a model created with Python and SciKit Learn from [this blog post](https://dev.to/azure/grab-your-wine-it-s-time-to-demystify-ml-and-nlp-47f7) to classify wine quality based on the description from a wine magazine. We are going to take that model, update it to use a [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and export it to an ONNX format. The reason we want to use ONNX format is because this is what will allow us to deploy it to many different platforms. There are many other benefits (such as performance) to using ONNX. Learn more about that [here](https://onnx.ai/). Since I ♥ C# we are going to use with the [onnxruntime nuget library](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) available for dotnet. However, if you prefer to use a different language many are supported.

### Prerquesites

- [Create a Free Azure Account!](https://azure.microsoft.com/en-us/free/?WT.mc.id=aiapril-devto-cassieb)
- [AML Azure Resource](https://docs.microsoft.com/en-us/azure/machine-learning/?WT.mc.id=aiapril-devto-cassieb) with a Notebook VM instance created
  - If you prefer to create the model locally I recommend downloading [Anaconda](https://www.anaconda.com/). However, this tutorial is written as if you are using AML.
- [VS Code](https://code.visualstudio.com/download)
- [.NET core 3.1](https://dotnet.microsoft.com/download)

### What is Open Neural Network Exchange (ONNX)?

Thanks for asking! ONNX is an open/common file format to enable you to use models with a variety of frameworks, tools, runtimes, and compilers. Once the model is exported to the ONNX format then use the ONNX Runtime a cross-platform, high performance scoring engine for ML models.

Build models in the Tensorflow, Keras, PyTorch, scikit-learn, CoreML, and other popular supported formats can be converted to the standard ONNX format, providing framework interoperability and helping to maximize the reach of hardware optimization.

### Why should you use it?

You are full of great questions. The answer is simple: it gives you the ability to use the same model and application code across different platforms. This means I can create this model in Python with SciKit Learn and use the resulting model in C#! Say whaaat? Yes, that is right. Save it to ONNX format then run it in C# with the onnxruntime!

## Create the Model with Azure Machine Learning

I have a model from the previous blog post to classify wine quality that we will use as the example model. See the note below if you have your own model you would like to use. Additionally, you should already have an Azure Machine Learning (AML) Studio created. If not, follow [these steps](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup#create-a-workspace) to create the workspace.

> [!NOTE]
> To use your own model visit the [ONNX Github tutorials](https://github.com/onnx/tutorials#converting-to-onnx-format) to see how to convert different frameworks and tools.

#### 1. Create Machine Learning Compute

- Click on the nav "Compute"
- Click "New"
- Enter a name for the resource
- Select "Machine Learning Compute" from the dropdown
- Select the machine size
- Enter the min and max nodes (recommend min of 0 and max of 5)
- Click "Create"
  ![Create Compute](https://globaleventcdn.blob.core.windows.net/assets/aiml/aiml30/CreateMlCompute.gif)

#### 2. Get the Jupyter Notebook

- Open JupyterLab for the compute instance in AML
- Click the Terminal to open a terminal tab
- Clone the github repo

```shell
git clone https://github.com/cassieview/onnx-csharp-serverless.git
```

- The `onnx-csharp-serverless` folder will appear. Navigate to the Jupyter Notebook `onnx-csharp-serverless/notebook/wine-nlp-onnx.ipynb`.

#### 3. Install the package

ONNX has different packages for python conversion to the ONNX format. Since we used SciKit Learn we will use the `skl2onnx` package to export our trained model

- In the terminal install the package with the following command

```python
pip install skl2onnx
```

#### 4. Run the code

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

#### 5. Save Model to Azure Storage

Now that we have exported the model into ONNX format lets save it to Azure Storage.

- Follow [these steps](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal?WT.mc.id=aiapril-devto-cassieb) to create a storage account and upload the model created.

## Deploy Model with Azure Functions

#### 1. Create Azure Function

- Install VS Code Azure Function extension:

  - Install the Azure Functions extension. You can use the Azure Functions extension to create and test functions and deploy them to Azure.
  - In Visual Studio Code, open Extensions and search for azure functions, or select this [link](vscode:extension/ms-azuretools.vscode-azurefunctions)
  - Select Install to install the extension for Visual Studio Code:

- Use VS Code to create our function using the command pallet.

  - Hit `CTRL-SHIFT-P` to open the command pallet
  - Type `create function` and select the create function option
  - From the popup select `Create new project` and create a folder for the project
  - When prompted for a language select C#. Note that you have many language options with functions!
  - Next select a template. We want an `HttpTrigger` and give it a name
  - Next it will prompt you for a namespace. I used Wine.Function but feel free to name it as you wish.
  - Access Rights are next, select `Function`
  - Select `open in current window`
  - You should be prompted in the bottom right corner to restore packages. If not you can always open the terminal and run `dotnet restore` to restore nuget packages.

- Run the project to validate its working

  - Hit `F5` to run project and test that its working
  - Once the function is up there will be a localhost endpoint displayed in the terminal output of VS Code. Paste that into a browser with a query string to test that it is working. The endpoint will look something like this:

  ```
  http://localhost:7071/api/wine?name=test
  ```

  - The result in the browser should look like this `Hello, test. This HTTP triggered function executed successfully`.
  - Stop the run.

#### 2. Install the Nuget Packages

Sweet, we now have an Azure Function ready to go. Lets install the nuget package we need to inference with our exported model in C#.

Open the terminal and run the below commands to install the ONNX package

```shell
dotnet add package Microsoft.ML.OnnxRuntime --version 1.2.0
dotnet add package System.Numerics.Tensors --version 0.1.0
```

Add the Azure Storage package with the below command

```shell
dotnet add package Azure.Storage.Blobs
```

Import the libraries at the top of the C# class.

```csharp
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
```

#### 3. Update the Code

Copy and paste the below code into the class created:

```csharp
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext context)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string review = req.Query["review"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            review = review ?? data?.review;

            var modelPath = GetFileAndPathFromStorage(context, "model327", "pipeline_quality.onnx");
            var inputTensor = new DenseTensor<string>(new string[] { review }, new int[] { 1, 1 });

            //create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            var session = new InferenceSession(modelPath);

            var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();
            var inferenceResult = output.First().AsDictionary<string, float>();

            return new JsonResult(inferenceResult);
        }
```

Add the helper method to the class below the `Run` method.

```csharp
 internal static string GetFileAndPathFromStorage(ExecutionContext context, string containerName, string fileName)
        {
            //Check if file already exists
            var filePath = System.IO.Path.Combine(context.FunctionDirectory, fileName);
            var fileExists = File.Exists(filePath);

            if (!fileExists)
            {
                //Get model from Azure Blob Storage.
                var connectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING");
                var blobServiceClient = new BlobServiceClient(connectionString);
                var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
                var blobClient = containerClient.GetBlobClient(fileName);

                // Download the blob's contents and save it to a file
                BlobDownloadInfo blobDownloadInfo = blobClient.Download();
                using (FileStream downloadFileStream = File.OpenWrite(filePath))
                {
                    blobDownloadInfo.Content.CopyTo(downloadFileStream);
                    downloadFileStream.Close();
                }

            }

            return filePath;
        }
```

Update the storage parameters

- Update the Storage Account connection in the `local.settings.json` to connect to your storage account. You will find the connection string in the created resource in Azure under `Access Keys`.
- Update the `containerName` and `fileName` to the names you used.

#### 4. Test the endpoint

Its time to test the function locally to make sure everything is working correctly.

- Hit `F5` to run project and test that its working
- This time we are going to use an actual wine review from [Wine Enthusiast](https://www.winemag.com/buying-guide/heitz-2014-marthas-vineyard-cabernet-sauvignon-napa-valley/). Rather than doing this through the browser we are going to use [Postman](https://www.postman.com/downloads/).
- Grab the endpoint from the terminal in vs code and paste it into a new tab in Postman.
- Change the `GET` dropdown to `POST`
- Select the `Body` tab
- Select the `raw` radiobutton
- check the `text` dropdown to `JSON`
- Paste the body into the text editor

```json
{
  "review": "From the famous Oakville site, this aged wine spends three years in 100% new French oak, one in neutral oak and an additional year in bottle. Though it has had time to evolve, it has years to go to unfurl its core of eucalyptus, mint and cedar. It shows an unmistakable crispness of red fruit, orange peel and stone, all honed by a grippy, generous palate."
}
```

- Hit send and you should see inference results.
  It should look list this:
  ![postman](\imgs\postman.PNG)

#### 5. Deploy the Endpoint to Azure

WOOHOO! We have created our model, the C# Azure Function, and tested it locally with Postman. Lets deploy it! The VS Code Azure Functions extension makes deploying to Azure quite simple. Follow [these steps](https://docs.microsoft.com/en-us/azure/azure-functions/functions-develop-vs-code?WT.mc.id=aiapril-devto-cassieb&tabs=csharp#publish-to-azure) to publish the function from VS Code.

Once the app is deployed we need update some application settings in the Function App.
- Navigate to the Function App in Azure Portal
- Select the name of the Function App you created
- Select `Configuration`
- Click `New Application setting` and add the storage connection string name and value.
- Click `Edit` on `WEBSITE_RUN_FROM_PACKAGE` and change the value to 0. This allows us to write files from our function.
- Save the changes
-NOTE: You may have to redeploy your function from VS Code after making this change.

# Resources

[ONNX Docs](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx?WT.mc.id=aiapril-devto-cassieb)
[Onnx C# API Docs](https://github.com/microsoft/onnxruntime/blob/master/docs/CSharp_API.md)
[Scikit learn pipeline onnx](http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html#l-example-tfidfvectorizer)
[ONNX Runtime Github](https://github.com/Microsoft/onnxruntime)
