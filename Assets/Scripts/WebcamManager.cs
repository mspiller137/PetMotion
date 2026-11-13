
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class WebcamManager : MonoBehaviour
{
    [Header("Webcam Settings")]
    [SerializeField] private int requestedWidth = 640;
    [SerializeField] private int requestedHeight = 480;
    [SerializeField] private int requestedFPS = 30;

    [Header("Display")]
    [SerializeField] private RawImage webcamDisplay;

    private WebCamTexture webcam;

    public WebCamTexture Webcam => webcam;

    private void Start()
    {
        StartCoroutine(InitializeWebcam());
    }

    private IEnumerator InitializeWebcam()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("No webcam detected!");
            yield break;
        }

        WebCamDevice device = WebCamTexture.devices[0];

        Debug.Log("Using webcam: " + device.name);

        webcam = new WebCamTexture(
            device.name,
            requestedWidth,
            requestedHeight,
            requestedFPS
        );

        webcam.Play();

        // Wait for webcam initialization.
        while (webcam.width <= 16)
        {
            yield return null;
        }

        if (webcamDisplay != null)
        {
            webcamDisplay.texture = webcam;
        }

        Debug.Log("Webcam initialized successfully!");
    }

    private void OnDisable()
    {
        if (webcam != null && webcam.isPlaying)
        {
            webcam.Stop();
        }
    }
}
