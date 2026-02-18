export const logger = {
  info: (category: string, ...args: any[]) => {
    console.log(`[${new Date().toISOString()}] [${category}]`, ...args);
  },
  warn: (category: string, ...args: any[]) => {
    console.warn(`[${new Date().toISOString()}] [${category}]`, ...args);
  },
  error: (category: string, ...args: any[]) => {
    console.error(`[${new Date().toISOString()}] [${category}]`, ...args);
  },
};
