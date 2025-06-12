// main.dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib; // For image processing

void main() {
  runApp(const DepthEstimationApp());
}

class DepthEstimationApp extends StatelessWidget {
  const DepthEstimationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Depth Estimation Demo',
      theme: ThemeData(
        primarySwatch: Colors.blueGrey,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Inter', // Using Inter font as per instructions
      ),
      home: const DepthEstimationHomePage(),
    );
  }
}

class DepthEstimationHomePage extends StatefulWidget {
  const DepthEstimationHomePage({super.key});

  @override
  State<DepthEstimationHomePage> createState() => _DepthEstimationHomePageState();
}

class _DepthEstimationHomePageState extends State<DepthEstimationHomePage> {
  File? _selectedImage; // Stores the original image picked by the user
  Uint8List? _depthMapImageBytes; // Stores the bytes of the processed depth map image
  Interpreter? _interpreter; // TensorFlow Lite interpreter
  bool _isProcessing = false; // To show loading indicator

  @override
  void initState() {
    super.initState();
    // _loadModel(); // Load the TFLite model when the app starts
  }

  // Function to load the TensorFlow Lite model
  Future<void> _loadModel() async {
    try {
      // Replace 'assets/depth_model.tflite' with the actual path to your model file.
      // Make sure you've added the model to your pubspec.yaml assets section.
      _interpreter = await Interpreter.fromAsset('assets/depth_model.tflite');
      print('Model loaded successfully!');
      print('Input Tensors: ${_interpreter?.getInputTensors()}');
      print('Output Tensors: ${_interpreter?.getOutputTensors()}');
    } catch (e) {
      print('Failed to load model: $e');
      // Optionally show an error dialog to the user
      _showErrorDialog('Failed to load AI model. Please ensure the model file is correct and accessible.');
    }
  }

  // Function to pick an image from the gallery
  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _depthMapImageBytes = null; // Clear previous depth map
      });
      // Automatically run depth estimation after picking an image
      _runDepthEstimation();
    }
  }

  // Placeholder function for running depth estimation using the loaded model
  Future<void> _runDepthEstimation() async {
    if (_selectedImage == null) {
      _showSnackbar('Please select an image first.');
      return;
    }
    if (_interpreter == null) {
      _showSnackbar('AI model not loaded. Cannot perform depth estimation.');
      return;
    }

    setState(() {
      _isProcessing = true; // Start showing loading indicator
    });

    try {
      // 1. Pre-process the image:
      //    - Read image bytes
      //    - Decode image
      //    - Resize to model's input size (e.g., 256x256, 512x512)
      //    - Normalize pixel values (e.g., to 0-1 or -1 to 1)
      //    - Convert to a format compatible with the model (e.g., Float32List, Int32List)

      final img_lib.Image? originalImage = img_lib.decodeImage(_selectedImage!.readAsBytesSync());
      if (originalImage == null) {
        _showErrorDialog('Failed to decode image.');
        setState(() { _isProcessing = false; });
        return;
      }

      // Get model input shape
      final inputShape = _interpreter!.getInputTensors()[0].shape;
      final inputType = _interpreter!.getInputTensors()[0].type;

      if (inputShape.length != 4) {
        _showErrorDialog('Unexpected model input shape. Expected [1, height, width, channels]. Got $inputShape');
        setState(() { _isProcessing = false; });
        return;
      }

      final int inputHeight = inputShape[1];
      final int inputWidth = inputShape[2];

      // Resize the image to the model's expected input dimensions
      final img_lib.Image resizedImage = img_lib.copyResize(
        originalImage,
        width: inputWidth,
        height: inputHeight,
      );

      // Prepare input tensor
      // This is a generic example. Your model might expect different normalization
      // or data types (e.g., Float32List for float models, Int32List for int models).
      // You need to adjust this based on your model's specific requirements.
      var input = List.generate(1, (i) => List.generate(inputHeight, (j) => List.generate(inputWidth, (k) => List.filled(3, 0.0))));

      if (inputType == TensorType.float32) {
        // Example for a float32 model (common for deep learning models)
        input = List.generate(1, (batch) {
          return List.generate(inputHeight, (y) {
            return List.generate(inputWidth, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                // img_lib.getRed(pixel) / 255.0,    // Normalize to 0-1
                // img_lib.getGreen(pixel) / 255.0,  // Normalize to 0-1
                // img_lib.getBlue(pixel) / 255.0,   // Normalize to 0-1
              ];
            });
          });
        });
      } else if (inputType == TensorType.uint8) {
        // Example for a uint8 model
        input = List.generate(1, (batch) {
          return List.generate(inputHeight, (y) {
            return List.generate(inputWidth, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                // img_lib.getRed(pixel),
                // img_lib.getGreen(pixel),
                // img_lib.getBlue(pixel),
              ];
            });
          });
        });
      } else {
        _showErrorDialog('Unsupported input tensor type: $inputType');
        setState(() { _isProcessing = false; });
        return;
      }

      // 2. Run inference:
      //    - Define output tensor structure based on your model's output
      //    - Run the interpreter
      final outputTensors = _interpreter!.getOutputTensors();
      if (outputTensors.isEmpty) {
        _showErrorDialog('Model has no output tensors defined.');
        setState(() { _isProcessing = false; });
        return;
      }

      final outputShape = outputTensors[0].shape;
      final outputType = outputTensors[0].type;

      // Initialize output buffer based on the expected output shape and type
      // This assumes a single output tensor. Adjust if your model has multiple outputs.
      dynamic output;
      if (outputType == TensorType.float32) {
        output = List.filled(outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);
      } else if (outputType == TensorType.uint8) {
        output = List.filled(outputShape.reduce((a, b) => a * b), 0).reshape(outputShape);
      } else {
        _showErrorDialog('Unsupported output tensor type: $outputType');
        setState(() { _isProcessing = false; });
        return;
      }

      _interpreter!.run(input, output);

      // 3. Post-process the output:
      //    - Convert the raw model output (e.g., depth values) into a visual image (e.g., grayscale depth map)
      //    - Normalize depth values to 0-255 for image display
      //    - Create an image from the processed data

      // Example post-processing for a grayscale depth map:
      // Assuming output is [1, height, width, 1] for a single channel depth map
      if (outputShape.length == 4 && outputShape[3] == 1) {
        final int outputHeight = outputShape[1];
        final int outputWidth = outputShape[2];

        // Find min and max depth values for normalization
        double minDepth = double.infinity;
        double maxDepth = double.negativeInfinity;

        for (int y = 0; y < outputHeight; y++) {
          for (int x = 0; x < outputWidth; x++) {
            final depthValue = output[0][y][x][0];
            if (depthValue < minDepth) minDepth = depthValue;
            if (depthValue > maxDepth) maxDepth = depthValue;
          }
        }

        // final depthImage = img_lib.Image(outputWidth, outputHeight);
        for (int y = 0; y < outputHeight; y++) {
          for (int x = 0; x < outputWidth; x++) {
            final depthValue = output[0][y][x][0];
            // Normalize depth to 0-255 and convert to grayscale pixel
            final normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);
            final pixelValue = (normalizedDepth * 255).round().clamp(0, 255);
            // depthImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
          }
        }

        setState(() {
          // _depthMapImageBytes = originalImage;
          // _depthMapImageBytes = Uint8List.fromList(img_lib.encodePng(depthImage));
        });
      } else {
        _showErrorDialog('Unexpected model output shape for depth map. Expected [1, height, width, 1]. Got $outputShape');
      }
    } catch (e) {
      print('Error during depth estimation: $e');
      _showErrorDialog('An error occurred during depth estimation: $e');
    } finally {
      setState(() {
        _isProcessing = false; // Stop loading indicator
      });
    }
  }

  // Helper to show a SnackBar message
  void _showSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  // Helper to show an error dialog
  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Error'),
          content: Text(message),
          actions: <Widget>[
            TextButton(
              child: const Text('OK'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  void dispose() {
    _interpreter?.close(); // Close the interpreter when the widget is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Depth Estimation Demo'),
        centerTitle: true,
      ),
      body: SingleChildScrollView( // Allows content to scroll if it overflows
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              // Section for Original Image
              Text(
                'Original Image',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blueGrey.shade200),
                ),
                height: 300, // Fixed height for image display
                alignment: Alignment.center,
                child: _selectedImage == null
                    ? Text(
                        'No image selected',
                        style: TextStyle(color: Colors.grey[600]),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.file(
                          _selectedImage!,
                          fit: BoxFit.contain, // Adjusts image to fit container
                          errorBuilder: (context, error, stackTrace) =>
                              const Text('Error loading image'),
                        ),
                      ),
              ),
              const SizedBox(height: 20),

              // Section for Depth Map
              Text(
                'Depth Map',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blueGrey.shade200),
                ),
                height: 300, // Fixed height for depth map display
                alignment: Alignment.center,
                child: _isProcessing
                    ? const CircularProgressIndicator() // Show loading indicator
                    : _depthMapImageBytes == null
                        ? Text(
                            _selectedImage == null ? 'Select an image to see depth map' : 'No depth map generated yet',
                            style: TextStyle(color: Colors.grey[600]),
                          )
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.memory(
                              _depthMapImageBytes!,
                              fit: BoxFit.contain,
                              errorBuilder: (context, error, stackTrace) =>
                                  const Text('Error loading depth map'),
                            ),
                          ),
              ),
              const SizedBox(height: 30),

              // Button to pick image
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: const Icon(Icons.photo_library),
                label: const Text('Pick Image from Gallery'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 10),
              // Button to run estimation (optional, as it runs automatically after pick)
              // This button can be useful for re-running if needed.
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _runDepthEstimation, // Disable when processing
                icon: _isProcessing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Icon(Icons.auto_awesome),
                label: Text(_isProcessing ? 'Processing...' : 'Run Depth Estimation'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(fontSize: 18),
                  backgroundColor: Colors.teal, // A different color for distinction
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
