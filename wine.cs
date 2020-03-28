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

            var modelPath = GetFileAndPathFromStorage(context, "model327", "pipeline_quality.onnx");
            var inputTensor = new DenseTensor<string>(new string[] { review }, new int[] { 1, 1 });

            //create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            var session = new InferenceSession(modelPath);

            var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();
            var inferenceResult = output.First().AsDictionary<string, float>();

            return new JsonResult(inferenceResult);
        }

        internal static string GetFileAndPathFromStorage(ExecutionContext context, string containerName, string fileName)
        {
            //Get model from Azure Blob Storage.
            var connectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING");
            var blobServiceClient = new BlobServiceClient(connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobClient = containerClient.GetBlobClient(fileName);
            var filePath = System.IO.Path.Combine(context.FunctionDirectory, fileName);

            // Download the blob's contents and save it to a file
            BlobDownloadInfo blobDownloadInfo = blobClient.Download();
            using (FileStream downloadFileStream = File.OpenWrite(filePath))
            {
                blobDownloadInfo.Content.CopyTo(downloadFileStream);
                downloadFileStream.Close();
            }

            return filePath;
        }
    }
}
