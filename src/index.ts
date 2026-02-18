import "dotenv/config";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "./logger";
import { CaptchaSolver } from "./solver";

// Load environment variables
const PORT = parseInt(process.env.PORT || "3000", 10);
const NODE_ENV = process.env.NODE_ENV || "development";

const app = new Hono();
const solver = new CaptchaSolver();

// CORS configuration
app.use(
  "/*",
  cors({
    origin: [
      "http://localhost:3000",
      "https://*.convex.cloud",
      "https://your-domain.com",
    ],
    credentials: true,
  }),
);

// Home route
app.get("/", (c) => {
  return c.json({
    message: "CAPTCHA Solver API",
    version: "1.0.0",
    endpoints: {
      health: "/health",
      solve: "POST /solve",
      batch: "POST /solve/batch",
      stats: "/stats",
    },
  });
});

// Health check
app.get("/health", (c) => {
  return c.json({
    status: "ok",
    ready: solver.isReady(),
    stats: solver.getStats(),
  });
});

// Solve CAPTCHA endpoint
app.post("/solve", async (c) => {
  try {
    const body = await c.req.json();
    const { image, id } = body;

    if (!image) {
      return c.json({ error: "Missing image data" }, 400);
    }

    const imageBuffer = Buffer.from(image, "base64");
    const result = await solver.solveCaptcha(imageBuffer, id);

    return c.json({
      success: result.success,
      code: result.code,
      confidence: result.confidence,
      method: result.method,
    });
  } catch (error) {
    logger.error("API", "Error solving CAPTCHA:", error);
    return c.json({ error: "Internal server error" }, 500);
  }
});

// Batch solve endpoint
app.post("/solve/batch", async (c) => {
  try {
    const body = await c.req.json();
    const { captchas } = body;

    if (!Array.isArray(captchas) || captchas.length === 0) {
      return c.json({ error: "Invalid captchas array" }, 400);
    }

    const results = await Promise.all(
      captchas.map(async ({ image, id }) => {
        const imageBuffer = Buffer.from(image, "base64");
        const result = await solver.solveCaptcha(imageBuffer, id);
        return {
          id,
          success: result.success,
          code: result.code,
          confidence: result.confidence,
        };
      }),
    );

    return c.json({ results });
  } catch (error) {
    logger.error("API", "Error solving batch CAPTCHAs:", error);
    return c.json({ error: "Internal server error" }, 500);
  }
});

// Stats endpoint
app.get("/stats", (c) => {
  return c.json(solver.getStats());
});

// Initialize solver
const initializeSolver = async () => {
  try {
    logger.info("SERVER", "Initializing CAPTCHA solver...");
    await solver.initialize(
      "./models/captcha_model.onnx",
      "./models/captcha_model_metadata.json",
    );
    logger.info("SERVER", "CAPTCHA solver initialized successfully");
  } catch (error) {
    logger.error("SERVER", "Failed to initialize solver:", error);
    process.exit(1);
  }
};

// Initialize solver on startup
await initializeSolver();

logger.info("SERVER", `Environment: ${NODE_ENV}`);
logger.info("SERVER", `🚀 CAPTCHA Solver API running on port ${PORT} (CPU only)`);

// Export for Bun server
export default {
  port: PORT,
  fetch: app.fetch,
};
