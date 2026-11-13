
using System.Collections;
using UnityEngine;
using Unity.InferenceEngine;

public class BlazeFaceInferenceTest : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] private ModelAsset faceModelAsset;

    [Header("Webcam")]
    [SerializeField] private WebcamManager webcamManager;

    private const int InputSize = 128;
    private const int DetectionCount = 896;

    private Worker worker;
    private Tensor<float> inputTensor;
    private Texture2D resizedTexture;

    private IEnumerator Start()
    {
        while (webcamManager == null ||
               webcamManager.Webcam == null ||
               webcamManager.Webcam.width <= 16)
        {
            yield return null;
        }

        if (faceModelAsset == null)
        {
            Debug.LogError("No BlazeFace model assigned!");
            yield break;
        }

        Model model = ModelLoader.Load(faceModelAsset);

        worker = new Worker(model, BackendType.CPU);

        inputTensor = new Tensor<float>(
            new TensorShape(1, 128, 128, 3)
        );

        resizedTexture = new Texture2D(
            InputSize,
            InputSize,
            TextureFormat.RGB24,
            false
        );

        Debug.Log("BlazeFace diagnostic started!");

        while (true)
        {
            RunDetection();
            yield return new WaitForSeconds(1f);
        }
    }

    private void RunDetection()
    {
        WebCamTexture webcam = webcamManager.Webcam;

        RenderTexture temporary = RenderTexture.GetTemporary(
            InputSize, InputSize, 0
        );

        Graphics.Blit(webcam, temporary);

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = temporary;

        resizedTexture.ReadPixels(
            new Rect(0, 0, InputSize, InputSize),
            0, 0
        );

        resizedTexture.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(temporary);

        Color32[] pixels = resizedTexture.GetPixels32();

        float pixelTotal = 0f;
        int pixelMinimum = 255;
        int pixelMaximum = 0;

        float inputMinimum = float.MaxValue;
        float inputMaximum = float.MinValue;
        float inputTotal = 0f;

        for (int i = 0; i < pixels.Length; i++)
        {
            Color32 pixel = pixels[i];

            pixelMinimum = Mathf.Min(
                pixelMinimum,
                Mathf.Min(pixel.r, Mathf.Min(pixel.g, pixel.b))
            );

            pixelMaximum = Mathf.Max(
                pixelMaximum,
                Mathf.Max(pixel.r, Mathf.Max(pixel.g, pixel.b))
            );

            pixelTotal += pixel.r + pixel.g + pixel.b;

            float r = pixel.r / 127.5f - 1f;
            float g = pixel.g / 127.5f - 1f;
            float b = pixel.b / 127.5f - 1f;

            int index = i * 3;

            inputTensor[index] = r;
            inputTensor[index + 1] = g;
            inputTensor[index + 2] = b;

            inputMinimum = Mathf.Min(
                inputMinimum, Mathf.Min(r, Mathf.Min(g, b))
            );

            inputMaximum = Mathf.Max(
                inputMaximum, Mathf.Max(r, Mathf.Max(g, b))
            );

            inputTotal += r + g + b;
        }

        float pixelAverage = pixelTotal / (pixels.Length * 3f);
        float inputAverage = inputTotal / (pixels.Length * 3f);

        Debug.Log(
            $"IMAGE: Min={pixelMinimum}, Max={pixelMaximum}, " +
            $"Average={pixelAverage:F2}, Centre={pixels[64 * 128 + 64]}"
        );

        Debug.Log(
            $"INPUT: Min={inputMinimum:F3}, " +
            $"Max={inputMaximum:F3}, Average={inputAverage:F3}"
        );

        worker.Schedule(inputTensor);

        Tensor<float> scores =
            worker.PeekOutput("classificators") as Tensor<float>;

        using Tensor<float> cpuScores = scores.ReadbackAndClone();

        float highestScore = float.MinValue;
        float lowestScore = float.MaxValue;

        for (int i = 0; i < DetectionCount; i++)
        {
            float score = cpuScores[i];

            highestScore = Mathf.Max(highestScore, score);
            lowestScore = Mathf.Min(lowestScore, score);
        }

        float confidence =
            1f / (1f + Mathf.Exp(-highestScore));

        Debug.Log(
            $"OUTPUT: Min={lowestScore:F3}, " +
            $"Max={highestScore:F3}, " +
            $"Confidence={confidence:P2}"
        );
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();

        if (resizedTexture != null)
        {
            Destroy(resizedTexture); 
        }
    }
}

