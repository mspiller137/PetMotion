
using UnityEngine;
using Unity.InferenceEngine;

public class BlazeFaceModelTest : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField] private ModelAsset faceModelAsset;

    private Model faceModel;

    private void Start()
    {
        InspectModel();
    }

    private void InspectModel()
    {
        if (faceModelAsset == null)
        {
            Debug.LogError("No BlazeFace model assigned!");
            return;
        }

        try
        {
            faceModel = ModelLoader.Load(faceModelAsset);

            Debug.Log("========== BLAZEFACE MODEL ==========");

            // Inspect model inputs.
            Debug.Log("INPUTS:");

            foreach (var input in faceModel.inputs)
            {
                Debug.Log(
                    "Name: " + input.name +
                    " | Shape: " + input.shape +
                    " | Data Type: " + input.dataType
                );
            }

            // Inspect model outputs.
            Debug.Log("OUTPUTS:");

            foreach (var output in faceModel.outputs)
            {
                Debug.Log("Output Name: " + output.name);
            }

            Debug.Log("====================================");
        }
        catch (System.Exception exception)
        {
            Debug.LogError(
                "Model inspection failed: " + exception
            );
        }
    }
}

