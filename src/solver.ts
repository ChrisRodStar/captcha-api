import { readFile } from "fs/promises";
import * as ort from "onnxruntime-node";
import sharp from "sharp";
import { logger } from "./logger";

interface ModelMetadata {
  input_shape: number[];
  chars: string;
  idx_to_char: Record<string, string>;
  normalization: {
    mean: number[];
    std: number[];
  };
}

interface SolveResult {
  code: string | null;
  success: boolean;
  method: "ONNX";
  confidence: number;
}

interface CaptchaSolverStats {
  totalAttempts: number;
  successfulDecodes: number;
  failures: number;
  averageConfidence: number;
}

export class CaptchaSolver {
  private session: ort.InferenceSession | null = null;
  private metadata: ModelMetadata | null = null;
  private isInitialized = false;

  private stats: CaptchaSolverStats = {
    totalAttempts: 0,
    successfulDecodes: 0,
    failures: 0,
    averageConfidence: 0,
  };

  async initialize(modelPath: string, metadataPath: string): Promise<void> {
    try {
      logger.info("SOLVER", `Loading ONNX model from: ${modelPath}`);

      // Configure CUDA execution provider for GPU acceleration
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: [
          {
            name: "cuda",
            deviceId: 0, // Use first GPU
          },
          "cpu", // Fallback to CPU if CUDA unavailable
        ],
        graphOptimizationLevel: "all",
        enableCpuMemArena: true,
        enableMemPattern: true,
      };

      this.session = await ort.InferenceSession.create(
        modelPath,
        sessionOptions,
      );

      // Log which execution provider is being used
      const provider = this.session.handler?.backend || "unknown";
      logger.info("SOLVER", `Using execution provider: ${provider}`);

      const metadataContent = await readFile(metadataPath, "utf-8");
      this.metadata = JSON.parse(metadataContent);

      // Test inference
      const inputShape = this.metadata!.input_shape;
      let height: number, width: number;

      if (inputShape.length === 3) {
        [, height, width] = inputShape;
      } else if (inputShape.length === 4) {
        [, , height, width] = inputShape;
      } else {
        throw new Error(`Unexpected input_shape length: ${inputShape.length}`);
      }

      const dummyData = new Float32Array(height * width).fill(0);
      const dummyTensor = new ort.Tensor("float32", dummyData, [
        1,
        1,
        height,
        width,
      ]);

      const inputName = this.session.inputNames[0];
      await this.session.run({ [inputName]: dummyTensor });

      this.isInitialized = true;
      logger.info("SOLVER", "ONNX model initialized successfully");
    } catch (error) {
      logger.error("SOLVER", "Failed to initialize:", error);
      throw error;
    }
  }

  private async preprocessImage(
    imageBuffer: Buffer,
  ): Promise<ort.Tensor | null> {
    try {
      if (!this.metadata) throw new Error("Metadata not loaded");

      const inputShape = this.metadata.input_shape;
      let height: number, width: number;

      if (inputShape.length === 3) {
        [, height, width] = inputShape;
      } else {
        [, , height, width] = inputShape;
      }

      const mean = this.metadata.normalization.mean[0];
      const std = this.metadata.normalization.std[0];

      const rawPixels = await sharp(imageBuffer)
        .grayscale()
        .resize(width, height, { fit: "fill", kernel: "lanczos3" })
        .raw()
        .toBuffer();

      const normalized = new Float32Array(rawPixels.length);
      for (let i = 0; i < rawPixels.length; i++) {
        normalized[i] = (rawPixels[i] / 255.0 - mean) / std;
      }

      return new ort.Tensor("float32", normalized, [1, 1, height, width]);
    } catch (error) {
      logger.error("SOLVER", "Error preprocessing image:", error);
      return null;
    }
  }

  async solveCaptcha(imageBuffer: Buffer, fid?: string): Promise<SolveResult> {
    if (!this.isInitialized || !this.session || !this.metadata) {
      return {
        code: null,
        success: false,
        method: "ONNX",
        confidence: 0.0,
      };
    }

    this.stats.totalAttempts++;
    const startTime = Date.now();

    try {
      const inputTensor = await this.preprocessImage(imageBuffer);
      if (!inputTensor) {
        this.stats.failures++;
        return { code: null, success: false, method: "ONNX", confidence: 0.0 };
      }

      const inputName = this.session.inputNames[0];
      const outputs = await this.session.run({ [inputName]: inputTensor });

      const idxToChar = this.metadata.idx_to_char;
      let predictedText = "";
      const confidences: number[] = [];

      for (let pos = 0; pos < 4; pos++) {
        const outputName = this.session.outputNames[pos];
        const outputTensor = outputs[outputName];
        const charProbs = outputTensor.data as Float32Array;

        let maxIdx = 0;
        let maxProb = charProbs[0];
        for (let i = 1; i < charProbs.length; i++) {
          if (charProbs[i] > maxProb) {
            maxProb = charProbs[i];
            maxIdx = i;
          }
        }

        predictedText += idxToChar[maxIdx.toString()];
        confidences.push(maxProb);
      }

      const avgConfidence =
        confidences.reduce((a, b) => a + b, 0) / confidences.length;
      const duration = Date.now() - startTime;

      const VALID_CHARACTERS = new Set(this.metadata.chars);
      const isValid =
        predictedText.length === 4 &&
        [...predictedText].every((c) => VALID_CHARACTERS.has(c));

      if (isValid) {
        this.stats.successfulDecodes++;
        this.stats.averageConfidence =
          (this.stats.averageConfidence * (this.stats.successfulDecodes - 1) +
            avgConfidence) /
          this.stats.successfulDecodes;

        logger.info(
          "SOLVER",
          `ID ${fid}: Solved '${predictedText}' (${avgConfidence.toFixed(3)}, ${duration}ms)`,
        );
        return {
          code: predictedText,
          success: true,
          method: "ONNX",
          confidence: avgConfidence,
        };
      } else {
        this.stats.failures++;
        logger.warn("SOLVER", `ID ${fid}: Failed validation`);
        return { code: null, success: false, method: "ONNX", confidence: 0.0 };
      }
    } catch (error) {
      this.stats.failures++;
      logger.error("SOLVER", `ID ${fid}: Exception:`, error);
      return { code: null, success: false, method: "ONNX", confidence: 0.0 };
    }
  }

  getStats(): CaptchaSolverStats {
    return { ...this.stats };
  }

  isReady(): boolean {
    return this.isInitialized;
  }
}
