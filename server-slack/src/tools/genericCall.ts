import { z } from "zod";
import { resolveSlackToken } from "../auth/token.js";
import { callSlack } from "../slack/client.js";

export const name = "slack.call";
export const description = "Call any Slack Web API method";

export const inputSchema = z.object({
  method: z.string(),
  params: z.record(z.any()).default({}),
  all: z.boolean().optional(),
  limit: z.number().int().optional(),
});

export type Input = z.infer<typeof inputSchema>;

export async function handler(ctx: any, input: Input) {
  const token = resolveSlackToken(ctx);
  return callSlack(token, input.method, input.params);
}
