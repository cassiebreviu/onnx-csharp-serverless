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

            string name = req.Query["name"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            name = name ?? data?.name;

            var modelPath = GetFilePathFromStorage(context, "model327", "pipeline_quality.onnx");

            // create tensor of string and shape
            var inputTensor = new DenseTensor<string>(new string[] { name }, new int[] { 1, 1 });


            //create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            Console.WriteLine(input);

            var session = new InferenceSession(modelPath);

            var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();
            var inferenceResult = output.First().AsDictionary<string, float>();

            return new JsonResult(inferenceResult);
        }

        internal static string GetFilePathFromStorage(ExecutionContext context, string containerName, string fileName)
        {
            //Get model from Azure Blob Storage.
            var connectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING");
            var blobServiceClient = new BlobServiceClient(connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobClient = containerClient.GetBlobClient(fileName);
            var filePath = System.IO.Path.Combine(context.FunctionDirectory, fileName);

            Console.WriteLine("\nDownloading blob to\n\t{0}\n", filePath);

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
