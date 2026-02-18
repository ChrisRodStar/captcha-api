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
  solveTimeMs: number;
}

interface CaptchaSolverStats {
  totalAttempts: number;
  successfulDecodes: number;
  failures: number;
  averageConfidence: number;
  averageSolveTimeMs: number;
  minSolveTimeMs: number;
  maxSolveTimeMs: number;
}

interface GpuStatus {
  enabled: boolean;
  available: boolean;
  provider: string | null;
  availableProviders: string[];
  requestedProviders: string[];
}

export class CaptchaSolver {
  private session: ort.InferenceSession | null = null;
  private metadata: ModelMetadata | null = null;
  private isInitialized = false;

  // Cached values for performance
  private height: number = 0;
  private width: number = 0;
  private mean: number = 0;
  private std: number = 0;
  private validCharacters: Set<string> = new Set();
  private inputName: string = "";

  // GPU status tracking
  private gpuStatus: GpuStatus = {
    enabled: false,
    available: false,
    provider: null,
    availableProviders: [],
    requestedProviders: [],
  };

  private stats: CaptchaSolverStats = {
    totalAttempts: 0,
    successfulDecodes: 0,
    failures: 0,
    averageConfidence: 0,
    averageSolveTimeMs: 0,
    minSolveTimeMs: Infinity,
    maxSolveTimeMs: 0,
  };

  async initialize(modelPath: string, metadataPath: string): Promise<void> {
    try {
      logger.info("SOLVER", `Loading ONNX model from: ${modelPath}`);

      // Determine execution providers - default to GPU enabled, can be disabled via USE_GPU=false
      const useGpu = process.env.USE_GPU !== "false";
      let executionProviders: string[] = [];
      let sessionCreated = false;

      if (useGpu) {
        // Try GPU providers first, fallback to CPU
        // CUDA for NVIDIA GPUs (Linux/Windows with CUDA installed)
        // DirectML for Windows with compatible GPU
        // Note: CUDA EP binaries are included by default in onnxruntime-node
        if (process.platform === "win32") {
          // Windows: Try DirectML first (works with AMD/NVIDIA/Intel GPUs), then CUDA, then CPU
          executionProviders = ["dml", "cuda", "cpu"];
        } else {
          // Linux: Try CUDA first, then CPU
          executionProviders = ["cuda", "cpu"];
        }
        logger.info(
          "SOLVER",
          "GPU support enabled - attempting to use GPU execution providers",
        );

        // Try to create session with GPU providers first
        try {
          const sessionOptions: ort.InferenceSession.SessionOptions = {
            executionProviders: executionProviders,
            graphOptimizationLevel: "all",
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionMode: "parallel",
            interOpNumThreads: 4,
            intraOpNumThreads: 4,
          };

          this.session = await ort.InferenceSession.create(
            modelPath,
            sessionOptions,
          );
          sessionCreated = true;
        } catch (gpuError) {
          // GPU initialization failed, fallback to CPU
          logger.warn(
            "SOLVER",
            "GPU initialization failed, falling back to CPU:",
            gpuError instanceof Error ? gpuError.message : String(gpuError),
          );
          executionProviders = ["cpu"];
          this.gpuStatus.enabled = false; // Mark as disabled since it failed
        }
      } else {
        // CPU-only execution
        executionProviders = ["cpu"];
        logger.info("SOLVER", "Using CPU-only execution");
      }

      // If session wasn't created (GPU failed), create with CPU
      if (!sessionCreated) {
        const sessionOptions: ort.InferenceSession.SessionOptions = {
          executionProviders: ["cpu"],
          graphOptimizationLevel: "all",
          enableCpuMemArena: true,
          enableMemPattern: true,
          executionMode: "parallel",
          interOpNumThreads: 4,
          intraOpNumThreads: 4,
        };

        this.session = await ort.InferenceSession.create(
          modelPath,
          sessionOptions,
        );
        // GPU failed, so we're using CPU
        this.gpuStatus.provider = "cpu";
        this.gpuStatus.available = false;
        this.gpuStatus.enabled = false; // Mark as disabled since it failed
        logger.info(
          "SOLVER",
          "Using execution provider: CPU (GPU unavailable)",
        );
      } else {
        // Session was created successfully - determine which provider is being used
        // Since we requested providers in order, ONNX Runtime uses the first available one
        // We can infer from whether GPU was requested and whether it succeeded
        if (useGpu && sessionCreated) {
          // Try to determine which provider was actually used
          // If we requested ["cuda", "cpu"] and it succeeded, it's likely CUDA
          // If we requested ["dml", "cuda", "cpu"] and it succeeded, it's likely DML or CUDA
          const gpuProviders = ["cuda", "dml", "tensorrt", "rocm"];
          const firstRequested = executionProviders[0];

          if (firstRequested && gpuProviders.includes(firstRequested)) {
            // Assume GPU provider is being used since session creation succeeded
            this.gpuStatus.provider = firstRequested;
            this.gpuStatus.available = true;
            logger.info(
              "SOLVER",
              `✅ GPU acceleration active using: ${firstRequested.toUpperCase()}`,
            );
          } else {
            // Fallback to CPU
            this.gpuStatus.provider = "cpu";
            this.gpuStatus.available = false;
            logger.info("SOLVER", "Using execution provider: CPU");
          }
        } else {
          // CPU-only mode
          this.gpuStatus.provider = "cpu";
          this.gpuStatus.available = false;
          logger.info("SOLVER", "Using execution provider: CPU");
        }
      }

      // Set status tracking
      this.gpuStatus.requestedProviders = executionProviders;
      this.gpuStatus.availableProviders = executionProviders; // Approximate - we don't have exact API
      if (this.gpuStatus.enabled === false && useGpu) {
        // Only update if not already set by error handler
        this.gpuStatus.enabled = useGpu;
      }

      logger.info("SOLVER", "ONNX session created successfully");

      const metadataContent = await readFile(metadataPath, "utf-8");
      this.metadata = JSON.parse(metadataContent);

      if (!this.session || !this.metadata) {
        throw new Error("Session or metadata initialization failed");
      }

      // Cache computed values for performance
      const inputShape = this.metadata.input_shape;

      if (inputShape.length === 3) {
        this.height = inputShape[1]!;
        this.width = inputShape[2]!;
      } else if (inputShape.length === 4) {
        this.height = inputShape[2]!;
        this.width = inputShape[3]!;
      } else {
        throw new Error(`Unexpected input_shape length: ${inputShape.length}`);
      }

      this.mean = this.metadata.normalization.mean[0]!;
      this.std = this.metadata.normalization.std[0]!;
      this.validCharacters = new Set(this.metadata.chars);
      this.inputName = this.session.inputNames[0]!;

      // Warm up the model with a test inference
      const dummyData = new Float32Array(this.height * this.width).fill(0);
      const dummyTensor = new ort.Tensor("float32", dummyData, [
        1,
        1,
        this.height,
        this.width,
      ]);

      await this.session.run({ [this.inputName]: dummyTensor });

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
      // Use cached values instead of recalculating
      const rawPixels = await sharp(imageBuffer)
        .grayscale()
        .resize(this.width, this.height, { fit: "fill", kernel: "linear" }) // linear is much faster than lanczos3
        .raw()
        .toBuffer();

      const normalized = new Float32Array(rawPixels.length);
      const invStd = 1.0 / this.std; // Multiply is faster than divide

      for (let i = 0; i < rawPixels.length; i++) {
        normalized[i] = (rawPixels[i]! / 255.0 - this.mean) * invStd;
      }

      return new ort.Tensor("float32", normalized, [
        1,
        1,
        this.height,
        this.width,
      ]);
    } catch (error) {
      logger.error("SOLVER", "Error preprocessing image:", error);
      return null;
    }
  }

  // Batch preprocessing for multiple images
  private async preprocessImages(
    imageBuffers: Buffer[],
  ): Promise<ort.Tensor[]> {
    return Promise.all(
      imageBuffers.map((buffer) => this.preprocessImage(buffer)),
    ).then((tensors) => tensors.filter((t): t is ort.Tensor => t !== null));
  }

  async solveCaptcha(imageBuffer: Buffer, fid?: string): Promise<SolveResult> {
    const startTime = performance.now();

    if (!this.isInitialized || !this.session || !this.metadata) {
      return {
        code: null,
        success: false,
        method: "ONNX",
        confidence: 0.0,
        solveTimeMs: 0,
      };
    }

    this.stats.totalAttempts++;

    try {
      const inputTensor = await this.preprocessImage(imageBuffer);
      if (!inputTensor) {
        this.stats.failures++;
        const solveTimeMs = performance.now() - startTime;
        return {
          code: null,
          success: false,
          method: "ONNX",
          confidence: 0.0,
          solveTimeMs,
        };
      }

      // Use cached input name
      const outputs = await this.session.run({
        [this.inputName]: inputTensor,
      });

      const idxToChar = this.metadata.idx_to_char;
      let predictedText = "";
      const confidences: number[] = [];

      for (let pos = 0; pos < 4; pos++) {
        const outputName = this.session.outputNames[pos]!;
        const outputTensor = outputs[outputName as string]!;
        const charProbs = outputTensor.data as Float32Array;

        let maxIdx = 0;
        let maxProb = charProbs[0]!;
        for (let i = 1; i < charProbs.length; i++) {
          if (charProbs[i]! > maxProb) {
            maxProb = charProbs[i]!;
            maxIdx = i;
          }
        }

        predictedText += idxToChar[maxIdx.toString()];
        confidences.push(maxProb);
      }

      const avgConfidence =
        confidences.reduce((a, b) => a + b, 0) / confidences.length;

      // Use cached valid characters set
      const isValid =
        predictedText.length === 4 &&
        [...predictedText].every((c) => this.validCharacters.has(c));

      const solveTimeMs = performance.now() - startTime;

      // Update timing stats
      this.stats.averageSolveTimeMs =
        (this.stats.averageSolveTimeMs * (this.stats.totalAttempts - 1) +
          solveTimeMs) /
        this.stats.totalAttempts;
      this.stats.minSolveTimeMs = Math.min(
        this.stats.minSolveTimeMs,
        solveTimeMs,
      );
      this.stats.maxSolveTimeMs = Math.max(
        this.stats.maxSolveTimeMs,
        solveTimeMs,
      );

      if (isValid) {
        this.stats.successfulDecodes++;
        this.stats.averageConfidence =
          (this.stats.averageConfidence * (this.stats.successfulDecodes - 1) +
            avgConfidence) /
          this.stats.successfulDecodes;

        // Only log in development or if confidence is low
        if (process.env.NODE_ENV === "development" || avgConfidence < 0.9) {
          logger.info(
            "SOLVER",
            `ID ${fid}: Solved '${predictedText}' (${avgConfidence.toFixed(3)}, ${solveTimeMs.toFixed(1)}ms)`,
          );
        }

        return {
          code: predictedText,
          success: true,
          method: "ONNX",
          confidence: avgConfidence,
          solveTimeMs,
        };
      } else {
        this.stats.failures++;
        logger.warn(
          "SOLVER",
          `ID ${fid}: Failed validation (${solveTimeMs.toFixed(1)}ms)`,
        );
        return {
          code: null,
          success: false,
          method: "ONNX",
          confidence: 0.0,
          solveTimeMs,
        };
      }
    } catch (error) {
      this.stats.failures++;
      const solveTimeMs = performance.now() - startTime;
      logger.error("SOLVER", `ID ${fid}: Exception:`, error);
      return {
        code: null,
        success: false,
        method: "ONNX",
        confidence: 0.0,
        solveTimeMs,
      };
    }
  }

  // Optimized batch solving - processes multiple captchas more efficiently
  async solveBatch(
    imageBuffers: Buffer[],
    ids?: string[],
  ): Promise<SolveResult[]> {
    if (!this.isInitialized || !this.session || !this.metadata) {
      return imageBuffers.map(() => ({
        code: null,
        success: false,
        method: "ONNX" as const,
        confidence: 0.0,
      }));
    }

    // Preprocess all images in parallel
    const tensors = await this.preprocessImages(imageBuffers);

    // Run inference on all tensors in parallel
    const results = await Promise.all(
      tensors.map(async (tensor, idx) => {
        const startTime = performance.now();
        const fid = ids?.[idx];
        this.stats.totalAttempts++;

        try {
          const outputs = await this.session!.run({
            [this.inputName]: tensor,
          });

          const idxToChar = this.metadata!.idx_to_char;
          let predictedText = "";
          const confidences: number[] = [];

          for (let pos = 0; pos < 4; pos++) {
            const outputName = this.session!.outputNames[pos]!;
            const outputTensor = outputs[outputName as string]!;
            const charProbs = outputTensor.data as Float32Array;

            let maxIdx = 0;
            let maxProb = charProbs[0]!;
            for (let i = 1; i < charProbs.length; i++) {
              if (charProbs[i]! > maxProb) {
                maxProb = charProbs[i]!;
                maxIdx = i;
              }
            }

            predictedText += idxToChar[maxIdx.toString()];
            confidences.push(maxProb);
          }

          const avgConfidence =
            confidences.reduce((a, b) => a + b, 0) / confidences.length;

          const isValid =
            predictedText.length === 4 &&
            [...predictedText].every((c) => this.validCharacters.has(c));

          const solveTimeMs = performance.now() - startTime;

          // Update timing stats
          this.stats.averageSolveTimeMs =
            (this.stats.averageSolveTimeMs * (this.stats.totalAttempts - 1) +
              solveTimeMs) /
            this.stats.totalAttempts;
          this.stats.minSolveTimeMs = Math.min(
            this.stats.minSolveTimeMs,
            solveTimeMs,
          );
          this.stats.maxSolveTimeMs = Math.max(
            this.stats.maxSolveTimeMs,
            solveTimeMs,
          );

          if (isValid) {
            this.stats.successfulDecodes++;
            this.stats.averageConfidence =
              (this.stats.averageConfidence *
                (this.stats.successfulDecodes - 1) +
                avgConfidence) /
              this.stats.successfulDecodes;

            return {
              code: predictedText,
              success: true,
              method: "ONNX" as const,
              confidence: avgConfidence,
              solveTimeMs,
            };
          } else {
            this.stats.failures++;
            return {
              code: null,
              success: false,
              method: "ONNX" as const,
              confidence: 0.0,
              solveTimeMs,
            };
          }
        } catch (error) {
          this.stats.failures++;
          const solveTimeMs = performance.now() - startTime;
          logger.error("SOLVER", `Batch ID ${fid}: Exception:`, error);
          return {
            code: null,
            success: false,
            method: "ONNX" as const,
            confidence: 0.0,
            solveTimeMs,
          };
        }
      }),
    );

    return results;
  }

  getStats(): CaptchaSolverStats {
    return { ...this.stats };
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  getGpuStatus(): GpuStatus {
    return { ...this.gpuStatus };
  }
}
