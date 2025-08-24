import { WebClient } from "@slack/web-api";
import { withRateLimitRetry } from "./rateLimit.js";

export function getSlackClient(token: string) {
  return new WebClient(token);
}

export async function callSlack(token: string, method: string, params: any) {
  const client = getSlackClient(token);
  const response = await withRateLimitRetry(() => client.apiCall(method, params));
  if (!response.ok) {
    const error: any = new Error(response.error || "Slack API error");
    throw error;
  }
  return response;
}
