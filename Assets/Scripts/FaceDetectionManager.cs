
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;

public class FaceDetectionManager : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] private ModelAsset faceModelAsset;

    [Header("Webcam")]
    [SerializeField] private WebcamManager webcamManager;

    [Header("UI")]
    [SerializeField] private RawImage webcamDisplay;
    [SerializeField] private RectTransform overlayRoot;
    [SerializeField] private Image faceBoxTemplate;

    [Header("Detection Settings")]
    [SerializeField, Range(0f, 1f)]
    private float confidenceThreshold = 0.5f;

    [SerializeField, Range(0f, 1f)]
    private float iouThreshold = 0.3f;

    [SerializeField, Range(0.05f, 1f)]
    private float detectionInterval = 0.1f;

    private const int InputSize = 128;
    private const int AnchorCount = 896;

    private Worker worker;
    private Tensor<float> inputTensor;
    private Texture2D resizedTexture;

    private Image faceBox;
    private float nextDetectionTime;

    private struct Detection
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public float confidence;
    }

    private IEnumerator Start()
    {
        while (webcamManager == null ||
               webcamManager.Webcam == null ||
               webcamManager.Webcam.width <= 16)
        {
            yield return null;
        }

        if (faceModelAsset == null ||
            overlayRoot == null ||
            faceBoxTemplate == null)
        {
            Debug.LogError(
                "FaceDetectionManager is missing Inspector references."
            );

            yield break;
        }

        Model model = ModelLoader.Load(faceModelAsset);

        worker = new Worker(model, BackendType.CPU);

        inputTensor = new Tensor<float>(
            new TensorShape(1, InputSize, InputSize, 3)
        );

        resizedTexture = new Texture2D(
            InputSize,
            InputSize,
            TextureFormat.RGB24,
            false
        );

        faceBox = Instantiate(faceBoxTemplate, overlayRoot);
        faceBox.name = "DetectedFaceBox";
        faceBox.gameObject.SetActive(false);

        Debug.Log("Face detection initialised!");
    }

    private void Update()
    {
        if (worker == null)
            return;

        if (Time.time < nextDetectionTime)
            return;

        nextDetectionTime = Time.time + detectionInterval;

        RunDetection();
    }

    private void RunDetection()
    {
        WebCamTexture webcam = webcamManager.Webcam;

        if (!webcam.didUpdateThisFrame)
            return;

        // Resize webcam image to model input dimensions.
        RenderTexture temporary = RenderTexture.GetTemporary(
            InputSize,
            InputSize,
            0
        );

        Graphics.Blit(webcam, temporary);

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = temporary;

        resizedTexture.ReadPixels(
            new Rect(0, 0, InputSize, InputSize),
            0,
            0
        );

        resizedTexture.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(temporary);

        // Convert RGB pixels to normalised model input.
        Color32[] pixels = resizedTexture.GetPixels32();

        for (int i = 0; i < pixels.Length; i++)
        {
            inputTensor[i * 3] =
                pixels[i].r / 127.5f - 1f;

            inputTensor[i * 3 + 1] =
                pixels[i].g / 127.5f - 1f;

            inputTensor[i * 3 + 2] =
                pixels[i].b / 127.5f - 1f;
        }

        // Run inference.
        worker.Schedule(inputTensor);

        Tensor<float> rawBoxes =
            worker.PeekOutput("regressors") as Tensor<float>;

        Tensor<float> rawScores =
            worker.PeekOutput("classificators") as Tensor<float>;

        using Tensor<float> boxes =
            rawBoxes.ReadbackAndClone();

        using Tensor<float> scores =
            rawScores.ReadbackAndClone();

        List<Detection> detections = new List<Detection>();

        // Decode all 896 anchor predictions.
        for (int i = 0; i < AnchorCount; i++)
        {
            float rawScore = scores[0, i, 0];

            // Convert raw score to confidence.
            float confidence =
                1f / (1f + Mathf.Exp(-rawScore));

            if (confidence < confidenceThreshold)
                continue;

            // Generate anchor centre.
            float anchorX;
            float anchorY;

            if (i < 512)
            {
                int cell = i / 2;

                anchorX = ((cell % 16) + 0.5f) / 16f;
                anchorY = ((cell / 16) + 0.5f) / 16f;
            }
            else
            {
                int cell = (i - 512) / 6;

                anchorX = ((cell % 8) + 0.5f) / 8f;
                anchorY = ((cell / 8) + 0.5f) / 8f;
            }

            // Decode box coordinates.
            float centerX =
                (boxes[0, i, 0] + anchorX * InputSize)
                / InputSize;

            float centerY =
                (boxes[0, i, 1] + anchorY * InputSize)
                / InputSize;

            float width =
                boxes[0, i, 2] / InputSize;

            float height =
                boxes[0, i, 3] / InputSize;

            if (width <= 0f || height <= 0f)
                continue;

            detections.Add(new Detection
            {
                x = centerX,
                y = centerY,
                width = width,
                height = height,
                confidence = confidence
            });
        }

        // Keep the strongest non-overlapping detection.
        detections.Sort(
            (a, b) => b.confidence.CompareTo(a.confidence)
        );

        Detection? bestDetection = null;

        foreach (Detection detection in detections)
        {
            if (bestDetection == null ||
                CalculateIoU(detection, bestDetection.Value)
                    < iouThreshold)
            {
                bestDetection = detection;
                break;
            }
        }

        if (bestDetection.HasValue)
        {
            DrawDetection(bestDetection.Value);
        }
        else
        {
            faceBox.gameObject.SetActive(false);
        }
    }

    private float CalculateIoU(Detection a, Detection b)
    {
        float aLeft = a.x - a.width / 2f;
        float aRight = a.x + a.width / 2f;
        float aBottom = a.y - a.height / 2f;
        float aTop = a.y + a.height / 2f;

        float bLeft = b.x - b.width / 2f;
        float bRight = b.x + b.width / 2f;
        float bBottom = b.y - b.height / 2f;
        float bTop = b.y + b.height / 2f;

        float intersectionWidth = Mathf.Max(
            0f,
            Mathf.Min(aRight, bRight) -
            Mathf.Max(aLeft, bLeft)
        );

        float intersectionHeight = Mathf.Max(
            0f,
            Mathf.Min(aTop, bTop) -
            Mathf.Max(aBottom, bBottom)
        );

        float intersection =
            intersectionWidth * intersectionHeight;

        float union =
            a.width * a.height +
            b.width * b.height -
            intersection;

        return union > 0f ? intersection / union : 0f;
    }


    private void DrawDetection(Detection detection)
    {
        // Horizontal coordinates remain unchanged.
        float left = detection.x - detection.width / 2f;
        float right = detection.x + detection.width / 2f;

        // Use the model's vertical coordinates directly.
        float bottom = detection.y - detection.height / 2f;
        float top = detection.y + detection.height / 2f;

        RectTransform rect = faceBox.rectTransform;

        rect.anchorMin = new Vector2(left, bottom);
        rect.anchorMax = new Vector2(right, top);

        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;

        faceBox.gameObject.SetActive(true);
    }


    private void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();

        if (resizedTexture != null)
            Destroy(resizedTexture);

        if (faceBox != null)
            Destroy(faceBox.gameObject);
    }
}
