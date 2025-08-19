function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withRateLimitRetry<T>(fn: () => Promise<T>, retries = 5): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: any) {
      const status = err?.statusCode || err?.code;
      if (status === 429 && attempt < retries) {
        const retryAfter = Number(err?.data?.retry_after || err?.headers?.["retry-after"] || 1);
        const wait = (retryAfter * 1000) + Math.random() * 1000;
        await sleep(wait);
        attempt++;
        continue;
      }
      throw err;
    }
  }
}
