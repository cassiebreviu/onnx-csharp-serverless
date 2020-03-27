using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;

namespace WineNlp.Function
{
    public static class wine
    {
        [FunctionName("wine")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string name = req.Query["name"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            name = name ?? data?.name;

            //get from model path
            //TODO: update from storage
            //TODO: add all models types here
            //string connectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING");
            var modelPath = @"C:\Code\onnx-csharp-serverless\pipeline_quality.onnx";
            // create tensor of string and shape
            var inputTensor = new DenseTensor<string>(new string[] { name }, new int[] { 1, 1 });


            //create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            Console.WriteLine(input);

            var session = new InferenceSession(modelPath, options);

            var outputProbList = session.Run(input).Last().Value as List<DisposableNamedOnnxValue>;
            var inferenceResult = outputProbList.First().Value as Dictionary<string, float>;

            string responseMessage = string.IsNullOrEmpty(name)
                ? "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response."
                : $"Hello, {name}. This HTTP triggered function executed successfully.";

            return new JsonResult(inferenceResult);
        }

        internal static object GetInstanceField(Type type, object instance, string fieldName)
        {
            FieldInfo field = type.GetField(fieldName);
            return field?.GetValue(instance);
        }
    }
}
