import { z } from "zod";
import { resolveSlackToken } from "../../auth/token.js";
import { callSlack } from "../../slack/client.js";

export const name = "slack.chat.postMessage";
export const description = "Post a message to a channel";

export const inputSchema = z.object({
  channel: z.string(),
  text: z.string(),
  thread_ts: z.string().optional(),
  blocks: z.any().optional(),
  mrkdwn: z.boolean().optional(),
  unfurl_links: z.boolean().optional(),
  unfurl_media: z.boolean().optional(),
});

export type Input = z.infer<typeof inputSchema>;

export async function handler(ctx: any, input: Input) {
  const token = resolveSlackToken(ctx);
  const params: any = {
    channel: input.channel,
    text: input.text,
  };
  if (input.thread_ts) params.thread_ts = input.thread_ts;
  if (input.blocks) params.blocks = input.blocks;
  if (input.mrkdwn !== undefined) params.mrkdwn = input.mrkdwn;
  if (input.unfurl_links !== undefined) params.unfurl_links = input.unfurl_links;
  if (input.unfurl_media !== undefined) params.unfurl_media = input.unfurl_media;
  const resp = await callSlack(token, "chat.postMessage", params);
  return resp;
}
