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
using System.Linq;
using System.Collections.Generic;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace WineNlp.Function
{
    public static class wine
    {
        [FunctionName("wine")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext context)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string review = req.Query["review"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            review = review ?? data?.review;

            var models = new Dictionary<string, string>();
            models.Add("points", GetFileAndPathFromStorage(context, "model327", "pipeline_points.onnx"));
            models.Add("price", GetFileAndPathFromStorage(context, "model327", "pipeline_price.onnx"));
            //models.Add("variety", GetFileAndPathFromStorage(context, "model327", "pipeline_variety.onnx"));

            var inputTensor = new DenseTensor<string>(new string[] { review }, new int[] { 1, 1 });
            //create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            //create now object points: result
            var inferenceResults = new Dictionary<string, IDictionary<Int64, float>>();

            foreach (var model in models)
            {
                var session = new InferenceSession(model.Value);
                var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();
                var blah = output.First();
                var inferenceResult = output.First().AsDictionary<Int64, float>();
                //var inferenceResult = output.First().AsDictionary<string, float>();//s.Value.ToString()
                var topFiveResult = inferenceResult.OrderByDescending(dict => dict.Value).Take(5)
                                    .ToDictionary(pair => pair.Key, pair => pair.Value);
                                    
                inferenceResults.Add(model.Key, topFiveResult);
                Console.Write(inferenceResult);
            }

            return new JsonResult(inferenceResults);
        }

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
    }
}
